import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Constants
TIME_STEPS = 128
NUM_CHANNELS = 1  # Using one EEG node (column)
BATCH_SIZE = 16
EPOCHS = 15
CLASS_MAP = {0: "Walking", 1: "Standing"}

# ========== Temporal Block (TCN) ==========
class TemporalBlock(tf.keras.Model):
    def __init__(self, out_channels, kernel_size, dilation_rate, dropout_rate):
        super().__init__()
        self.conv1 = layers.Conv1D(out_channels, kernel_size, padding='causal', dilation_rate=dilation_rate)
        self.relu1 = layers.ReLU()
        self.dropout1 = layers.Dropout(dropout_rate)

        self.conv2 = layers.Conv1D(out_channels, kernel_size, padding='causal', dilation_rate=dilation_rate)
        self.relu2 = layers.ReLU()
        self.dropout2 = layers.Dropout(dropout_rate)

        self.downsample = None
        self.final_relu = layers.ReLU()
        self.out_channels = out_channels

    def build(self, input_shape):
        input_channels = input_shape[-1]
        if input_channels != self.out_channels:
            self.downsample = layers.Conv1D(self.out_channels, 1)
        else:
            self.downsample = lambda x: x
        super().build(input_shape)

    def call(self, x, training=False):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out, training=training)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out, training=training)

        res = self.downsample(x)
        return self.final_relu(out + res)

class TemporalConvNet(tf.keras.Model):
    def __init__(self, num_channels, kernel_size=3, dropout_rate=0.2):
        super().__init__()
        self.blocks = []
        for i, out_channels in enumerate(num_channels):
            dilation_rate = 2 ** i
            self.blocks.append(TemporalBlock(out_channels, kernel_size, dilation_rate, dropout_rate))

    def call(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)
        return x

# ========== CNN + TCN + BiLSTM ==========
class NeuroFlexModel(tf.keras.Model):
    def __init__(self, num_channels, time_steps, num_classes):
        super().__init__()

        self.spatial_conv = tf.keras.Sequential([
            layers.Conv2D(16, (1, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (1, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
        ])

        self.tcn = TemporalConvNet(num_channels=[64, 64], kernel_size=3, dropout_rate=0.2)
        self.rnn = layers.Bidirectional(layers.LSTM(128))

        self.dense = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training=False):
        # x shape: (batch, channels, time_steps)
        x = tf.expand_dims(x, -1)  # (batch, channels, time_steps, 1)
        x = self.spatial_conv(x)   # (batch, channels, time_steps, 32)
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, time_steps, channels, 32)
        shape = tf.shape(x)
        x = tf.reshape(x, (shape[0], shape[1], shape[2] * shape[3]))  # (batch, time_steps, channels*32)
        x = self.tcn(x, training=training)                           # TCN output
        x = self.rnn(x)                                              # BiLSTM output
        return self.dense(x)

# ========== Data Loader for Training ==========
def load_training_data(walk_csv, stand_csv):
    walk_df = pd.read_csv(walk_csv)
    stand_df = pd.read_csv(stand_csv)

    walk_data = walk_df.iloc[:, 0].values.astype(np.float32)
    stand_data = stand_df.iloc[:, 0].values.astype(np.float32)

    print(f"Original walk data length: {len(walk_data)}")
    print(f"Original stand data length: {len(stand_data)}")

    scaler = StandardScaler()
    walk_data = scaler.fit_transform(walk_data.reshape(-1,1)).flatten()
    stand_data = scaler.transform(stand_data.reshape(-1,1)).flatten()

    num_walk_windows = len(walk_data) // TIME_STEPS
    num_stand_windows = len(stand_data) // TIME_STEPS

    walk_data = walk_data[:num_walk_windows * TIME_STEPS]
    stand_data = stand_data[:num_stand_windows * TIME_STEPS]

    print(f"Trimmed walk data length: {len(walk_data)}")
    print(f"Trimmed stand data length: {len(stand_data)}")

    walk_windows = walk_data.reshape(num_walk_windows, TIME_STEPS, NUM_CHANNELS)
    stand_windows = stand_data.reshape(num_stand_windows, TIME_STEPS, NUM_CHANNELS)

    walk_labels = np.zeros(num_walk_windows, dtype=np.int32)
    stand_labels = np.ones(num_stand_windows, dtype=np.int32)

    X = np.concatenate((walk_windows, stand_windows), axis=0)
    y = np.concatenate((walk_labels, stand_labels), axis=0)

    print(f"Total training windows: {X.shape[0]}")

    return X, y

# ========== Data Loader for Testing ==========
def load_test_data(test_csv):
    test_df = pd.read_csv(test_csv)
    test_data = test_df.iloc[:, 0].values.astype(np.float32)

    scaler = StandardScaler()
    test_data = scaler.fit_transform(test_data.reshape(-1,1)).flatten()

    num_windows = len(test_data) // TIME_STEPS
    test_data = test_data[:num_windows * TIME_STEPS]

    test_windows = test_data.reshape(num_windows, TIME_STEPS, NUM_CHANNELS)
    return test_windows

# ========== Train Model ==========
def train_model(walk_csv, stand_csv):
    print(" Loading and preparing training data...")
    X, y = load_training_data(walk_csv, stand_csv)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = NeuroFlexModel(num_channels=NUM_CHANNELS, time_steps=TIME_STEPS, num_classes=2)
    model.build(input_shape=(None, NUM_CHANNELS, TIME_STEPS))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(" Starting training...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

    print(" Training complete. Evaluating on validation data...")
    loss, acc = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {acc * 100:.2f}%")

    return model

# ========== Prediction ==========
def predict(model, test_csv):
    print(" Loading test data...")
    X_test = load_test_data(test_csv)

    print(" Predicting on test data...")
    preds = model.predict(X_test)
    pred_labels = np.argmax(preds, axis=1)

    print(" Predictions (0=Walking, 1=Standing):")
    print(pred_labels)

if __name__ == "__main__":
    walk_csv = r"C:\Users\eeshd\Downloads\cortex-example-master\cortex-example-master\python\EEGTEST\walking.csv"
    stand_csv = r"C:\Users\eeshd\Downloads\cortex-example-master\cortex-example-master\python\EEGTEST\standing.csv"
    test_csv = r"C:\Users\eeshd\Downloads\cortex-example-master\cortex-example-master\python\EEGTEST\Test.csv"

    model = train_model(walk_csv, stand_csv)
    predict(model, test_csv)
