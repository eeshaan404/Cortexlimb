import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from scipy.interpolate import CubicSpline
import odrive
from odrive.enums import *
import time

use_dual_odrive = input("Activate dual ODrive control mode? (yes/no): ").strip().lower()

if use_dual_odrive == 'yes':
    print("Dual ODrive control mode activated.")

    odrive_serial_number_1 = '395E347A3331'
    odrive_serial_number_2 = '395E347A1234'

    odrv1 = odrive.find_any(serial_number=odrive_serial_number_1)
    odrv2 = odrive.find_any(serial_number=odrive_serial_number_2)

    axis1 = odrv1.axis0
    axis2 = odrv2.axis0

    axis1.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
    axis2.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL

    axis1.trap_traj.config.vel_limit = 3.6
    axis1.trap_traj.config.accel_limit = 1.8
    axis1.trap_traj.config.decel_limit = 1.8

    axis2.trap_traj.config.vel_limit = 3.6
    axis2.trap_traj.config.accel_limit = 1.8
    axis2.trap_traj.config.decel_limit = 1.8

    axis1.controller.config.input_mode = INPUT_MODE_TRAP_TRAJ
    axis2.controller.config.input_mode = INPUT_MODE_TRAP_TRAJ

    positions1 = [0, 0, 0]
    positions2 = [0, 0.5, 0]

    while True:
        for pos1 in positions1:
            axis1.controller.input_pos = pos1
            time.sleep(1)
        for pos2 in positions2:
            axis2.controller.input_pos = pos2
            time.sleep(1)
        axis1.controller.input_pos = 0
        axis2.controller.input_pos = 0
        time.sleep(1)

else:
    print("Running AI model with ODrive velocity control...")

    def load_data(file_path):
        data = np.genfromtxt(file_path, delimiter=',')
        return data

    def normalize_data(data):
        if np.any(np.isnan(data)):
            print("Warning: NaNs found in data!")
            data = np.nan_to_num(data)
        scaler = MinMaxScaler()
        return scaler.fit_transform(data), scaler

    train_data_path = r"C:\\Users\\eeshd\\Downloads\\alphatest.txt"
    test_data_path = r"C:\\Users\\eeshd\\Downloads\\Alpha waves test.txt"

    train_data, train_scaler = normalize_data(load_data(train_data_path))
    test_data, test_scaler = normalize_data(load_data(test_data_path))

    def create_timeseries_dataset(data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length, -1])
        return np.array(X), np.array(y)

    sequence_length = 60
    X_train, y_train = create_timeseries_dataset(train_data, sequence_length)
    X_test, y_test = create_timeseries_dataset(test_data, sequence_length)

    def create_cnn_model(input_shape):
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(32, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        return model

    def custom_accuracy(y_true, y_pred):
        tolerance = 0.1
        return tf.reduce_mean(tf.cast(tf.abs(y_true - y_pred) <= tolerance, tf.float32))

    input_shape = (sequence_length, train_data.shape[1])
    cnn_model = create_cnn_model(input_shape)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=3,
        decay_rate=0.7,
        staircase=True
    )

    optimizer = Adam(learning_rate=lr_schedule, clipvalue=1.0)
    cnn_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', custom_accuracy])
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    cnn_features_train = cnn_model.predict(X_train)
    cnn_features_test = cnn_model.predict(X_test)

    def create_tft_model(input_shape):
        inputs = Input(shape=input_shape)
        lstm_out = LSTM(128, return_sequences=True)(inputs)
        attention_out = LSTM(128)(lstm_out)
        dense_out = Dense(64, activation='relu')(attention_out)
        scaled_output = Dense(1, activation='sigmoid')(dense_out)
        outputs = layers.Lambda(lambda x: 0.5 + x * 0.5)(scaled_output)
        model = Model(inputs, outputs)
        return model

    input_shape = (cnn_features_train.shape[1], 1)
    tft_model = create_tft_model(input_shape)
    tft_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    tft_model.fit(cnn_features_train, y_train, epochs=50, batch_size=32, validation_data=(cnn_features_test, y_test))

    predicted_speeds = tft_model.predict(cnn_features_test)
    print("Predicted walking speeds:", predicted_speeds)

    def cubic_spline_interpolation(predicted_speeds):
        time_steps = np.arange(len(predicted_speeds))
        cs = CubicSpline(time_steps, predicted_speeds.flatten())
        return cs

    cs = cubic_spline_interpolation(predicted_speeds)

    def speed_to_velocity(speed, time_interval=1):
        if len(speed) < 2:
            raise ValueError("Not enough data to compute velocity.")
        return np.gradient(speed, time_interval)

    velocity = speed_to_velocity(cs(np.arange(len(predicted_speeds))))

    def control_odrive_motor(velocity, cs):
        print("Connecting to ODrive...")
        odrv0 = odrive.find_any()
        print("ODrive connected!")
        odrv0.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        odrv0.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL

        for i, v in enumerate(velocity):
            target_angle = cs(i)
            odrv0.axis0.controller.input_vel = v
            current_position = odrv0.axis0.encoder.pos_estimate
            print(f"Applying velocity: {v:.3f} (Target angle: {target_angle:.3f}) - Position: {current_position:.3f}")

    control_odrive_motor(velocity, cs)