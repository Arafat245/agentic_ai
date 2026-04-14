"""
Phase 1: Data Pipeline & Feature Extraction
- Loads labels and aligns 16-second sensor windows (ACC, PPG, Light)
- Extracts hand-crafted features from each window
- Saves processed feature matrix for downstream ML
"""

import pickle
import pandas as pd
import numpy as np
from scipy import stats, signal
from pathlib import Path
import warnings
import time

warnings.filterwarnings('ignore')

PROJECT_DIR = Path('/mnt/sdb/arafat/agentic_ai/project')
DATA_DIR = PROJECT_DIR / 'Processed data'
LABELS_FILE = PROJECT_DIR / 'labels_for_non_acoustic_model.pkl'
OUTPUT_DIR = PROJECT_DIR / 'temp_exps'
WINDOW_SECONDS = 16


def load_labels():
    labels = pickle.load(open(LABELS_FILE, 'rb'))
    labels['Sensor_start_time'] = pd.to_datetime(labels['Sensor_start_time'], format='ISO8601')
    labels['category'] = labels['category'].astype(int)
    return labels


def load_sensor_data(subject_id):
    """Load all 3 sensor files for a subject, return dict of dataframes."""
    base = DATA_DIR / subject_id
    sensors = {}

    for name, fname, value_cols in [
        ('acc', 'Smartwatch_AccelerometerDatum.pkl', ['X', 'Y', 'Z']),
        ('ppg', 'Smartwatch_PPG_Health_SDK.pkl', ['PPG Green']),
        ('light', 'Smartwatch_LightDatum.pkl', ['Light']),
    ]:
        fpath = base / fname
        if fpath.exists():
            df = pickle.load(open(fpath, 'rb'))
            df = df[value_cols + ['T']].copy()
            df = df.sort_values('T').reset_index(drop=True)
            sensors[name] = df
        else:
            sensors[name] = None

    return sensors


def extract_window(sensor_df, start_time, window_sec=WINDOW_SECONDS):
    """Extract sensor data within [start_time, start_time + window_sec]."""
    if sensor_df is None:
        return None
    end_time = start_time + pd.Timedelta(seconds=window_sec)
    mask = (sensor_df['T'] >= start_time) & (sensor_df['T'] < end_time)
    window = sensor_df.loc[mask]
    if len(window) < 3:
        return None
    return window


# --- Feature Extraction Functions ---

def time_domain_features(values, prefix):
    """Extract time-domain features from a 1D signal."""
    feats = {}
    if len(values) < 3:
        return {f'{prefix}_{k}': np.nan for k in [
            'mean', 'std', 'min', 'max', 'median', 'rms', 'iqr',
            'skewness', 'kurtosis', 'zero_crossing_rate', 'peak_amplitude',
            'range'
        ]}

    feats[f'{prefix}_mean'] = np.mean(values)
    feats[f'{prefix}_std'] = np.std(values)
    feats[f'{prefix}_min'] = np.min(values)
    feats[f'{prefix}_max'] = np.max(values)
    feats[f'{prefix}_median'] = np.median(values)
    feats[f'{prefix}_rms'] = np.sqrt(np.mean(values ** 2))
    feats[f'{prefix}_iqr'] = np.percentile(values, 75) - np.percentile(values, 25)
    feats[f'{prefix}_skewness'] = stats.skew(values)
    feats[f'{prefix}_kurtosis'] = stats.kurtosis(values)
    feats[f'{prefix}_range'] = np.max(values) - np.min(values)
    feats[f'{prefix}_peak_amplitude'] = np.max(np.abs(values))

    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(values - np.mean(values))) != 0)
    feats[f'{prefix}_zero_crossing_rate'] = zero_crossings / len(values)

    return feats


def freq_domain_features(values, sampling_rate, prefix):
    """Extract frequency-domain features from a 1D signal."""
    feats = {}
    if len(values) < 8 or sampling_rate <= 0:
        return {f'{prefix}_{k}': np.nan for k in [
            'dominant_freq', 'spectral_energy', 'spectral_entropy',
            'band_power_low', 'band_power_mid', 'band_power_high'
        ]}

    n = len(values)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    fft_vals = np.abs(np.fft.rfft(values - np.mean(values)))
    power = fft_vals ** 2

    if np.sum(power) == 0:
        return {f'{prefix}_{k}': 0.0 for k in [
            'dominant_freq', 'spectral_energy', 'spectral_entropy',
            'band_power_low', 'band_power_mid', 'band_power_high'
        ]}

    feats[f'{prefix}_dominant_freq'] = freqs[np.argmax(power[1:]) + 1] if len(power) > 1 else 0.0
    feats[f'{prefix}_spectral_energy'] = np.sum(power)

    # Spectral entropy
    psd_norm = power / np.sum(power)
    psd_norm = psd_norm[psd_norm > 0]
    feats[f'{prefix}_spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm))

    # Band powers (low: 0-2Hz, mid: 2-5Hz, high: 5+Hz)
    max_freq = sampling_rate / 2
    low_mask = freqs <= min(2.0, max_freq)
    mid_mask = (freqs > 2.0) & (freqs <= min(5.0, max_freq))
    high_mask = freqs > min(5.0, max_freq)
    total_power = np.sum(power) + 1e-10
    feats[f'{prefix}_band_power_low'] = np.sum(power[low_mask]) / total_power
    feats[f'{prefix}_band_power_mid'] = np.sum(power[mid_mask]) / total_power if np.any(mid_mask) else 0.0
    feats[f'{prefix}_band_power_high'] = np.sum(power[high_mask]) / total_power if np.any(high_mask) else 0.0

    return feats


def extract_acc_features(window):
    """Extract features from accelerometer window."""
    if window is None:
        return {f'acc_{ax}_{ft}': np.nan
                for ax in ['x', 'y', 'z', 'mag']
                for ft in ['mean', 'std', 'min', 'max', 'median', 'rms', 'iqr',
                           'skewness', 'kurtosis', 'zero_crossing_rate', 'peak_amplitude',
                           'range', 'dominant_freq', 'spectral_energy', 'spectral_entropy',
                           'band_power_low', 'band_power_mid', 'band_power_high']}

    feats = {}
    x, y, z = window['X'].values, window['Y'].values, window['Z'].values
    mag = np.sqrt(x**2 + y**2 + z**2)

    # Estimate sampling rate from data
    t_diff = window['T'].diff().dt.total_seconds().dropna()
    sr = 1.0 / t_diff.median() if len(t_diff) > 0 and t_diff.median() > 0 else 18.5

    for arr, name in [(x, 'acc_x'), (y, 'acc_y'), (z, 'acc_z'), (mag, 'acc_mag')]:
        feats.update(time_domain_features(arr, name))
        feats.update(freq_domain_features(arr, sr, name))

    # Cross-axis correlations
    if len(x) >= 3:
        feats['acc_corr_xy'] = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else 0.0
        feats['acc_corr_xz'] = np.corrcoef(x, z)[0, 1] if np.std(x) > 0 and np.std(z) > 0 else 0.0
        feats['acc_corr_yz'] = np.corrcoef(y, z)[0, 1] if np.std(y) > 0 and np.std(z) > 0 else 0.0
    else:
        feats['acc_corr_xy'] = np.nan
        feats['acc_corr_xz'] = np.nan
        feats['acc_corr_yz'] = np.nan

    # Signal Magnitude Area
    feats['acc_sma'] = (np.sum(np.abs(x)) + np.sum(np.abs(y)) + np.sum(np.abs(z))) / len(x)

    return feats


def extract_ppg_features(window):
    """Extract features from PPG window."""
    prefix_list = ['ppg']
    if window is None:
        return {f'ppg_{ft}': np.nan
                for ft in ['mean', 'std', 'min', 'max', 'median', 'rms', 'iqr',
                           'skewness', 'kurtosis', 'zero_crossing_rate', 'peak_amplitude',
                           'range', 'dominant_freq', 'spectral_energy', 'spectral_entropy',
                           'band_power_low', 'band_power_mid', 'band_power_high']}

    values = window['PPG Green'].values.astype(float)
    t_diff = window['T'].diff().dt.total_seconds().dropna()
    sr = 1.0 / t_diff.median() if len(t_diff) > 0 and t_diff.median() > 0 else 25.0

    feats = {}
    feats.update(time_domain_features(values, 'ppg'))
    feats.update(freq_domain_features(values, sr, 'ppg'))

    return feats


def extract_light_features(window):
    """Extract features from Light window."""
    if window is None:
        return {f'light_{ft}': np.nan
                for ft in ['mean', 'std', 'min', 'max', 'median', 'rms', 'iqr',
                           'skewness', 'kurtosis', 'zero_crossing_rate', 'peak_amplitude',
                           'range', 'slope']}

    values = window['Light'].values.astype(float)
    feats = time_domain_features(values, 'light')

    # Slope/trend
    if len(values) >= 3:
        t = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(t, values)
        feats['light_slope'] = slope
    else:
        feats['light_slope'] = np.nan

    return feats


def extract_all_features(acc_window, ppg_window, light_window):
    """Extract all features from all sensor windows."""
    feats = {}
    feats.update(extract_acc_features(acc_window))
    feats.update(extract_ppg_features(ppg_window))
    feats.update(extract_light_features(light_window))
    return feats


def process_all_subjects():
    """Main pipeline: load data, extract features for all samples."""
    labels = load_labels()
    subjects = sorted(labels['P_ID'].unique())

    print(f"Processing {len(labels)} samples across {len(subjects)} subjects...")
    print(f"Window size: {WINDOW_SECONDS} seconds")
    print()

    all_features = []
    missing_counts = {'acc': 0, 'ppg': 0, 'light': 0, 'total_skipped': 0}

    for i, subject in enumerate(subjects):
        t0 = time.time()
        subject_labels = labels[labels['P_ID'] == subject].reset_index(drop=True)
        sensors = load_sensor_data(subject)

        subject_features = []
        for _, row in subject_labels.iterrows():
            start_time = row['Sensor_start_time']

            acc_win = extract_window(sensors['acc'], start_time)
            ppg_win = extract_window(sensors['ppg'], start_time)
            light_win = extract_window(sensors['light'], start_time)

            if acc_win is None:
                missing_counts['acc'] += 1
            if ppg_win is None:
                missing_counts['ppg'] += 1
            if light_win is None:
                missing_counts['light'] += 1

            feats = extract_all_features(acc_win, ppg_win, light_win)
            feats['P_ID'] = row['P_ID']
            feats['category'] = row['category']
            feats['Sensor_start_time'] = str(row['Sensor_start_time'])
            subject_features.append(feats)

        all_features.extend(subject_features)
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(subjects)}] {subject}: {len(subject_labels)} samples, {elapsed:.1f}s")

    df = pd.DataFrame(all_features)

    print(f"\nTotal samples: {len(df)}")
    print(f"Total features: {len([c for c in df.columns if c not in ['P_ID', 'category', 'Sensor_start_time']])}")
    print(f"Missing windows - ACC: {missing_counts['acc']}, PPG: {missing_counts['ppg']}, Light: {missing_counts['light']}")

    # Count rows with any NaN in feature columns
    feat_cols = [c for c in df.columns if c not in ['P_ID', 'category', 'Sensor_start_time']]
    nan_rows = df[feat_cols].isna().any(axis=1).sum()
    print(f"Rows with any NaN features: {nan_rows}")

    # Save
    output_path = OUTPUT_DIR / 'features_16s.pkl'
    df.to_pickle(output_path)
    print(f"\nSaved to {output_path}")

    return df


if __name__ == '__main__':
    df = process_all_subjects()
    print("\nFeature columns:")
    feat_cols = [c for c in df.columns if c not in ['P_ID', 'category', 'Sensor_start_time']]
    for c in feat_cols:
        print(f"  {c}")
    print(f"\nClass distribution:")
    print(df['category'].value_counts().sort_index())
