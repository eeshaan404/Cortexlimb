import numpy as np
import ast
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Step 1: Read the Data (same as in your provided code)
file_path = '/home/shiva/Downloads/5-33_5-34(walking)/Alpha1.log'  # Replace with your actual file path
sensor_values = []

with open(file_path, 'r') as f:
    for line in f:
        try:
            numbers = ast.literal_eval(line.strip())
            sensor_values.append(numbers)
        except ValueError as e:
            print(f"Skipping invalid line: {line.strip()}")
            continue

sensor_values = np.array(sensor_values)

# Step 2: Normalize the Data
scaler = StandardScaler()
for i in range(sensor_values.shape[1]):
    sensor_values[:, i] = scaler.fit_transform(sensor_values[:, i].reshape(-1, 1)).flatten()

# Step 3: Add Statistical Features
mean_values = np.mean(sensor_values, axis=1)
std_values = np.std(sensor_values, axis=1)
max_values = np.max(sensor_values, axis=1)
min_values = np.min(sensor_values, axis=1)
sensor_values_extended = np.column_stack((sensor_values, mean_values, std_values, max_values, min_values))

sensor_values_extended = sensor_values_extended.reshape((sensor_values_extended.shape[0], sensor_values_extended.shape[1], 1))

# Step 4: Generate Labels (dummy labels for demonstration)
labels = np.random.randint(0, 3, len(sensor_values_extended))  # Dummy labels: 0 = Slow, 1 = Medium, 2 = Fast

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(sensor_values_extended, labels, test_size=0.2, random_state=42)

# Step 6: Compute Class Weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Step 7: Build the Temporal Fusion Transformer (TFT) Model
def build_tft_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    selected_features = layers.Dense(64, activation='relu')(inputs)
    x = layers.Conv1D(64, kernel_size=3, padding='causal', activation='relu')(selected_features)
    x = layers.MaxPooling1D(pool_size=2)(x)
    grn_1 = layers.GRU(64, return_sequences=True)(x)
    grn_2 = layers.GRU(64)(grn_1)
    attention = layers.Attention()([grn_2, grn_1])
    attention = layers.GlobalAveragePooling1D()(attention)
    outputs = layers.Dense(3, activation='softmax')(attention)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Initialize and compile the TFT model
model = build_tft_model((X_train.shape[1], X_train.shape[2]))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 8: Train the Model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weight_dict)

# Step 9: Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Step 10: Predict Speed for Each Sample
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
