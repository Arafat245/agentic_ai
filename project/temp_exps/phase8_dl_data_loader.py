"""
Phase 8 — Data loader for raw sensor time series.
Loads 16-second windows from Accelerometer, PPG, and Light sensors.
Resamples to fixed length and creates PyTorch datasets for TCN/LSTM/Transformer.
"""

import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.signal import resample
import warnings

warnings.filterwarnings('ignore')

PROJECT_DIR = Path('/mnt/sdb/arafat/agentic_ai/project')
DATA_DIR = PROJECT_DIR / 'Processed data'
LABELS_FILE = PROJECT_DIR / 'labels_for_non_acoustic_model.pkl'

# Target lengths after resampling (16 seconds)
# ACC ~20Hz × 16s = 320, PPG 25Hz × 16s = 400, Light 5Hz × 16s = 80
# Unify to a common length for simplicity
TARGET_LENGTH = 320  # Resample all to 320 timesteps (20Hz equivalent)
WINDOW_SECONDS = 16

SEED = 42


def load_labels():
    with open(LABELS_FILE, 'rb') as f:
        labels = pickle.load(f)
    labels['category'] = labels['category'].astype(int)
    labels['Sensor_start_time'] = pd.to_datetime(labels['Sensor_start_time'], format='ISO8601')
    return labels


def load_sensor(subject_id, sensor_name, value_cols):
    """Load a sensor pkl file for a subject."""
    path = DATA_DIR / subject_id / sensor_name
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        df = pickle.load(f)
    df = df[value_cols + ['T']].copy()
    df['T'] = pd.to_datetime(df['T'], format='ISO8601')
    df = df.sort_values('T').reset_index(drop=True)
    return df


def extract_window(sensor_df, start_time, end_time, value_cols, target_length=TARGET_LENGTH):
    """Extract and resample a time window from sensor data."""
    if sensor_df is None:
        return np.zeros((target_length, len(value_cols)))

    mask = (sensor_df['T'] >= start_time) & (sensor_df['T'] < end_time)
    window = sensor_df.loc[mask, value_cols].values

    if len(window) < 3:
        return np.zeros((target_length, len(value_cols)))

    # Resample to target length
    if len(window) != target_length:
        resampled = np.zeros((target_length, window.shape[1]))
        for ch in range(window.shape[1]):
            resampled[:, ch] = resample(window[:, ch], target_length)
        return resampled

    return window


def build_dataset_for_subject(subject_id, labels_df):
    """Load all sensor data and extract windows for one subject."""
    subj_labels = labels_df[labels_df['P_ID'] == subject_id].copy()

    # Load sensor data
    acc_df = load_sensor(subject_id, 'Smartwatch_AccelerometerDatum.pkl', ['X', 'Y', 'Z'])
    ppg_df = load_sensor(subject_id, 'Smartwatch_PPG_Health_SDK.pkl', ['PPG Green'])
    light_df = load_sensor(subject_id, 'Smartwatch_LightDatum.pkl', ['Light'])

    X_list = []
    y_list = []

    for _, row in subj_labels.iterrows():
        start = row['Sensor_start_time']
        end = start + pd.Timedelta(seconds=WINDOW_SECONDS)

        # Extract windows
        acc_win = extract_window(acc_df, start, end, ['X', 'Y', 'Z'], TARGET_LENGTH)    # (320, 3)
        ppg_win = extract_window(ppg_df, start, end, ['PPG Green'], TARGET_LENGTH)       # (320, 1)
        light_win = extract_window(light_df, start, end, ['Light'], TARGET_LENGTH)       # (320, 1)

        # Concatenate: (320, 5) — 3 acc + 1 ppg + 1 light
        combined = np.concatenate([acc_win, ppg_win, light_win], axis=1)
        X_list.append(combined)
        y_list.append(row['category'])

    X = np.array(X_list, dtype=np.float32)  # (n_samples, 320, 5)
    y = np.array(y_list, dtype=np.int64)
    return X, y


class SensorDataset(Dataset):
    """PyTorch dataset for sensor time series."""
    def __init__(self, X, y):
        # X: (n, seq_len, channels) -> (n, channels, seq_len) for Conv1d
        self.X = torch.FloatTensor(X).permute(0, 2, 1)  # (n, 5, 320)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def normalize_per_channel(X_train, X_test):
    """Z-score normalize per channel using training statistics."""
    # X shape: (n, seq_len, channels)
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std[std < 1e-8] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def load_all_subjects():
    """Load and cache all subjects' raw sensor windows."""
    labels = load_labels()
    subjects = sorted(labels['P_ID'].unique())

    cache_path = PROJECT_DIR / 'temp_exps' / 'raw_sensor_windows.pkl'
    if cache_path.exists():
        print(f"Loading cached sensor windows from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"Extracting raw sensor windows for {len(subjects)} subjects...")
    all_data = {}
    for i, subj in enumerate(subjects):
        X, y = build_dataset_for_subject(subj, labels)
        all_data[subj] = (X, y)
        print(f"  [{i+1}/{len(subjects)}] {subj}: {X.shape[0]} samples, "
              f"{y.sum()} pos, {len(y)-y.sum()} neg", flush=True)

    with open(cache_path, 'wb') as f:
        pickle.dump(all_data, f)
    print(f"Cached to {cache_path}")
    return all_data


if __name__ == '__main__':
    data = load_all_subjects()
    total = sum(len(y) for _, (_, y) in data.items())
    print(f"\nTotal: {total} samples across {len(data)} subjects")
    # Show shape of first subject
    subj = list(data.keys())[0]
    X, y = data[subj]
    print(f"Example: {subj} — X={X.shape}, y={y.shape}")
