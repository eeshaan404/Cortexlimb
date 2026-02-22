import numpy as np
import asyncio
import websockets
import json
from scipy.signal import butter, lfilter
from collections import deque
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D,
    LSTM, Dense, Dropout, Add
)
from tensorflow.keras.optimizers import Adam
import os
import time
import odrive
from odrive.enums import *

MOTION_TYPES = ["walking", "standing", "climbing_stairs", "moving_leg", "stationary"]
MODEL_PATH = "trained_tft_model.h5"
LABEL_MAP = {
    0: "standing",
    1: "walking",
    2: "moving_leg",
    3: "climbing_stairs"
}

# -------------------------------
# Build CNN-RNN Hybrid Model
# -------------------------------
def build_deep_cnn_rnn_model(timesteps=100, num_features=2, num_classes=4):
    input_layer = Input(shape=(timesteps, num_features))
    x = input_layer
    for _ in range(40):
        y = Conv1D(64, 3, padding='same', activation='relu')(x)
        y = BatchNormalization()(y)
        y = Conv1D(64, 3, padding='same', activation='relu')(y)
        x = Add()([x, y])
        x = MaxPooling1D(pool_size=2)(x)
    for _ in range(99):
        x = LSTM(64, return_sequences=True)(x)
    x = LSTM(128)(x)
    x = Dropout(0.5)(x)
    for _ in range(8):
        x = Dense(256, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------
# Bandpass Filter
# -------------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype='band')

def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=128, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

# -------------------------------
# Build TFT Model
# -------------------------------
def build_tft_model(timesteps=20, ik_features=5):
    ik_input = Input(shape=(timesteps, ik_features), name="ik_input")
    x = LSTM(128, return_sequences=True)(ik_input)
    x = LSTM(128)(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='linear', name="predicted_knee_angle")(x)
    model = Model(inputs=ik_input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# -------------------------------
# ODrive and Trajectory Control
# -------------------------------
def move_odrive_to_angle_trajectory(odrive_axis, angle_deg, duration=0.3):
    TURNS_PER_DEGREE = 1 / 360
    turns = angle_deg * TURNS_PER_DEGREE
    odrive_axis.controller.config.input_mode = INPUT_MODE_TRAP_TRAJ
    odrive_axis.trap_traj.config.vel_limit = 2
    odrive_axis.trap_traj.config.accel_limit = 5
    odrive_axis.trap_traj.config.decel_limit = 5
    odrive_axis.controller.input_pos = turns
    print(f"[ODRIVE] Moving to {angle_deg:.2f} degrees using trajectory mode.")
    time.sleep(duration)

def generate_knee_trajectory(max_angle, smooth=True):
    return [0, max_angle * 0.6, max_angle, max_angle * 0.6, 0] if smooth else [0, max_angle, 0]

def send_trajectory_sequence(odrive_axis, trajectory, delay=0.3):
    for angle in trajectory:
        move_odrive_to_angle_trajectory(odrive_axis, angle, duration=delay)

# -------------------------------
# Runtime TFT Prediction
# -------------------------------
def run_live_inference(motion_label):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("TFT model not found. Please train it first.")
    model = load_model(MODEL_PATH)
    path = f"kinematics_data/{motion_label}.npy"
    if not os.path.exists(path):
        raise FileNotFoundError(f"IK data for {motion_label} missing.")
    ik_data = np.load(path)
    ik_sample = ik_data[0:1]
    predicted_knee_angle = model.predict(ik_sample, verbose=0)[0][0]
    print(f"\n[PREDICTION] Motion: {motion_label}, Max Knee Angle: {predicted_knee_angle:.2f} degrees")
    odrv = odrive.find_any()
    trajectory = generate_knee_trajectory(predicted_knee_angle)
    send_trajectory_sequence(odrv.axis0, trajectory)

# -------------------------------
# Real-Time EEG Motion Detection + Actuation
# -------------------------------
async def emotiv_stream(cnn_rnn_model):
    window_size = 100
    fs = 128
    buffer_F3 = deque(maxlen=window_size)
    buffer_F4 = deque(maxlen=window_size)
    async with websockets.connect("wss://localhost:6868", ssl=None) as ws:
        await ws.send(json.dumps({
            "jsonrpc": "2.0", "method": "authorize",
            "params": {
                "clientId": "your_client_id",
                "clientSecret": "your_client_secret",
                "license": "your_license",
                "debit": 1
            }, "id": 1
        }))
        token = json.loads(await ws.recv())['result']['cortexToken']
        await ws.send(json.dumps({
            "jsonrpc": "2.0", "method": "createSession",
            "params": {"cortexToken": token, "headset": "EPOCPLUS-XXXXXX", "status": "active"},
            "id": 2
        }))
        session_id = json.loads(await ws.recv())['result']['id']
        await ws.send(json.dumps({
            "jsonrpc": "2.0", "method": "subscribe",
            "params": {"cortexToken": token, "session": session_id, "streams": ["eeg"]},
            "id": 3
        }))
        await ws.recv()
        print("\n[EEG] Streaming F3/F4...")
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if 'eeg' in data:
                eeg = data['eeg']
                f3 = eeg[3]
                f4 = eeg[12]
                buffer_F3.append(f3)
                buffer_F4.append(f4)
                if len(buffer_F3) == window_size:
                    eeg_window = np.array([list(buffer_F3), list(buffer_F4)]).T
                    f3_filt = bandpass_filter(eeg_window[:, 0], fs=fs)
                    f4_filt = bandpass_filter(eeg_window[:, 1], fs=fs)
                    window = np.stack([f3_filt, f4_filt], axis=1).reshape(1, window_size, 2)
                    pred = cnn_rnn_model.predict(window, verbose=0)
                    pred_class = np.argmax(pred)
                    label = LABEL_MAP.get(pred_class, "unknown")
                    print(f"[EEG] Prediction: {label} | Confidence: {pred[0][pred_class]:.2f}")
                    run_live_inference(label)

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    print("\n[INFO] Initializing CNN-RNN Hybrid + TFT actuator system...")
    cnn_rnn_model = build_deep_cnn_rnn_model()
    asyncio.run(emotiv_stream(cnn_rnn_model))
