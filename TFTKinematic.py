import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
import os
import time
import odrive
from odrive.enums import *
import joblib

MOTION_TYPES = ["walking", "standing", "climbing_stairs", "moving_leg", "stationary"]
MODEL_PATH = "trained_tft_model.h5"

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
# Trajectory Control for ODrive
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

# -------------------------------
# Generate Knee Trajectory
# -------------------------------
def generate_knee_trajectory(max_angle, smooth=True):
    if smooth:
        return [0, max_angle * 0.6, max_angle, max_angle * 0.6, 0]
    else:
        return [0, max_angle, 0]

def send_trajectory_sequence(odrive_axis, trajectory, delay=0.3):
    for angle in trajectory:
        move_odrive_to_angle_trajectory(odrive_axis, angle, duration=delay)

# -------------------------------
# Load Training Dataset
# -------------------------------
def load_training_dataset():
    ik_sequences = []
    angle_labels = []
    for motion in MOTION_TYPES:
        path = f"kinematics_data/{motion}.npy"
        if os.path.exists(path):
            data = np.load(path)
            for seq in data:
                ik_sequences.append(seq)
                angle_labels.append(np.max(seq[:, 0]))  # assuming angle is in column 0
        else:
            print(f"[WARN] Missing: {path}")
    return np.array(ik_sequences), np.array(angle_labels)

# -------------------------------
# Train the TFT Model
# -------------------------------
def train_tft_model():
    print("\n[TRAINING] Loading training data...")
    X, y = load_training_dataset()
    model = build_tft_model()
    model.fit(X, y, epochs=100, batch_size=16)
    model.save(MODEL_PATH)
    print("[TRAINING] Model saved to", MODEL_PATH)

# -------------------------------
# Runtime Inference
# -------------------------------
def run_live_inference(motion_label):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("TFT model not found. Please train it first.")
    model = load_model(MODEL_PATH)
    path = f"kinematics_data/{motion_label}.npy"
    if not os.path.exists(path):
        raise FileNotFoundError(f"IK data for {motion_label} missing.")

    ik_data = np.load(path)
    ik_sample = ik_data[0:1]  # Take the first sample
    predicted_knee_angle = model.predict(ik_sample, verbose=0)[0][0]
    print(f"\n[PREDICTION] Motion: {motion_label}, Max Knee Angle: {predicted_knee_angle:.2f} degrees")

    print("[ODRIVE] Connecting...")
    odrv = odrive.find_any()
    trajectory = generate_knee_trajectory(predicted_knee_angle)
    send_trajectory_sequence(odrv.axis0, trajectory)
    print("✅ Motion execution complete.")

# -------------------------------
# Main Control
# -------------------------------
if __name__ == "__main__":
    mode = input("Enter mode (train / run): ").strip().lower()
    if mode == "train":
        train_tft_model()
    elif mode == "run":
        label = input("Enter detected motion label (walking, standing, etc.): ").strip().lower()
        run_live_inference(label)
    else:
        print("Invalid mode. Use 'train' or 'run'.")
