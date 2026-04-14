"""
Feature Extraction for Social Interaction Detection
- Uses first 16 seconds of each 90-second sensor window
- PPG features via NeuroKit2 (HRV, peak detection, signal quality)
- Accelerometer: time-domain, frequency-domain, cross-axis stats
- Light: time-domain + trend features
- Saves per-subject CSV files and a combined CSV
"""

import pickle
import pandas as pd
import numpy as np
from scipy import stats, signal
from pathlib import Path
import warnings
import time
import neurokit2 as nk

warnings.filterwarnings('ignore')

PROJECT_DIR = Path('/mnt/sdb/arafat/agentic_ai/project')
DATA_DIR = PROJECT_DIR / 'Processed data'
LABELS_FILE = PROJECT_DIR / 'labels_for_non_acoustic_model.pkl'
OUTPUT_DIR = PROJECT_DIR / 'temp_exps'
WINDOW_SECONDS = 16


# ===================== DATA LOADING =====================

def load_labels():
    labels = pickle.load(open(LABELS_FILE, 'rb'))
    labels['Sensor_start_time'] = pd.to_datetime(labels['Sensor_start_time'], format='ISO8601')
    labels['category'] = labels['category'].astype(int)
    return labels


def load_sensor_data(subject_id):
    """Load all 3 sensor files for a subject."""
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
    """Extract sensor data within [start_time, start_time + 16s)."""
    if sensor_df is None:
        return None
    end_time = start_time + pd.Timedelta(seconds=window_sec)
    mask = (sensor_df['T'] >= start_time) & (sensor_df['T'] < end_time)
    window = sensor_df.loc[mask]
    if len(window) < 5:
        return None
    return window


# ===================== HELPER: SAFE STAT =====================

def safe_stat(func, arr, default=np.nan):
    """Compute a statistic safely, returning default on failure."""
    try:
        val = func(arr)
        return val if np.isfinite(val) else default
    except Exception:
        return default


# ===================== ACCELEROMETER FEATURES =====================

def extract_acc_features(window):
    """Extract time-domain, frequency-domain, and cross-axis features from accelerometer."""
    nan_feats = {}
    axes = ['x', 'y', 'z', 'mag']
    time_keys = ['mean', 'std', 'min', 'max', 'median', 'rms', 'iqr',
                 'skew', 'kurtosis', 'zcr', 'peak_amp', 'range',
                 'energy', 'p25', 'p75']
    freq_keys = ['dom_freq', 'spectral_energy', 'spectral_entropy',
                 'band_low', 'band_mid', 'band_high', 'spectral_centroid']
    for ax in axes:
        for k in time_keys + freq_keys:
            nan_feats[f'acc_{ax}_{k}'] = np.nan
    for k in ['corr_xy', 'corr_xz', 'corr_yz', 'sma', 'jerk_mean', 'jerk_std']:
        nan_feats[f'acc_{k}'] = np.nan

    if window is None:
        return nan_feats

    x, y, z = window['X'].values, window['Y'].values, window['Z'].values
    mag = np.sqrt(x**2 + y**2 + z**2)

    # Estimate sampling rate
    t_diff = window['T'].diff().dt.total_seconds().dropna()
    sr = 1.0 / t_diff.median() if len(t_diff) > 0 and t_diff.median() > 0 else 18.5

    feats = {}
    for arr, name in [(x, 'acc_x'), (y, 'acc_y'), (z, 'acc_z'), (mag, 'acc_mag')]:
        # Time-domain
        feats[f'{name}_mean'] = safe_stat(np.mean, arr)
        feats[f'{name}_std'] = safe_stat(np.std, arr)
        feats[f'{name}_min'] = safe_stat(np.min, arr)
        feats[f'{name}_max'] = safe_stat(np.max, arr)
        feats[f'{name}_median'] = safe_stat(np.median, arr)
        feats[f'{name}_rms'] = safe_stat(lambda a: np.sqrt(np.mean(a**2)), arr)
        feats[f'{name}_iqr'] = np.percentile(arr, 75) - np.percentile(arr, 25)
        feats[f'{name}_skew'] = safe_stat(stats.skew, arr)
        feats[f'{name}_kurtosis'] = safe_stat(stats.kurtosis, arr)
        feats[f'{name}_range'] = np.max(arr) - np.min(arr)
        feats[f'{name}_peak_amp'] = np.max(np.abs(arr))
        feats[f'{name}_energy'] = safe_stat(lambda a: np.sum(a**2) / len(a), arr)
        feats[f'{name}_p25'] = np.percentile(arr, 25)
        feats[f'{name}_p75'] = np.percentile(arr, 75)

        # Zero crossing rate
        centered = arr - np.mean(arr)
        zcr = np.sum(np.diff(np.sign(centered)) != 0) / len(arr)
        feats[f'{name}_zcr'] = zcr

        # Frequency-domain
        if len(arr) >= 8:
            n = len(arr)
            freqs = np.fft.rfftfreq(n, d=1.0 / sr)
            fft_vals = np.abs(np.fft.rfft(arr - np.mean(arr)))
            power = fft_vals ** 2
            total_power = np.sum(power) + 1e-10

            feats[f'{name}_dom_freq'] = freqs[np.argmax(power[1:]) + 1] if len(power) > 1 else 0.0
            feats[f'{name}_spectral_energy'] = np.sum(power)

            psd_norm = power / total_power
            psd_norm = psd_norm[psd_norm > 0]
            feats[f'{name}_spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm))

            feats[f'{name}_spectral_centroid'] = np.sum(freqs * power) / total_power

            max_freq = sr / 2
            low_mask = freqs <= min(2.0, max_freq)
            mid_mask = (freqs > 2.0) & (freqs <= min(5.0, max_freq))
            high_mask = freqs > min(5.0, max_freq)
            feats[f'{name}_band_low'] = np.sum(power[low_mask]) / total_power
            feats[f'{name}_band_mid'] = np.sum(power[mid_mask]) / total_power if np.any(mid_mask) else 0.0
            feats[f'{name}_band_high'] = np.sum(power[high_mask]) / total_power if np.any(high_mask) else 0.0
        else:
            for k in freq_keys:
                feats[f'{name}_{k}'] = np.nan

    # Cross-axis correlations
    if np.std(x) > 0 and np.std(y) > 0:
        feats['acc_corr_xy'] = np.corrcoef(x, y)[0, 1]
    else:
        feats['acc_corr_xy'] = 0.0
    if np.std(x) > 0 and np.std(z) > 0:
        feats['acc_corr_xz'] = np.corrcoef(x, z)[0, 1]
    else:
        feats['acc_corr_xz'] = 0.0
    if np.std(y) > 0 and np.std(z) > 0:
        feats['acc_corr_yz'] = np.corrcoef(y, z)[0, 1]
    else:
        feats['acc_corr_yz'] = 0.0

    # Signal Magnitude Area
    feats['acc_sma'] = (np.sum(np.abs(x)) + np.sum(np.abs(y)) + np.sum(np.abs(z))) / len(x)

    # Jerk (derivative of magnitude)
    if len(mag) >= 5:
        jerk = np.diff(mag) * sr
        feats['acc_jerk_mean'] = np.mean(np.abs(jerk))
        feats['acc_jerk_std'] = np.std(jerk)
    else:
        feats['acc_jerk_mean'] = np.nan
        feats['acc_jerk_std'] = np.nan

    return feats


# ===================== PPG FEATURES (NeuroKit2) =====================

def extract_ppg_features(window):
    """Extract PPG features using NeuroKit2: HRV, peak stats, signal quality, time/freq domain."""
    # Define all PPG feature keys for NaN fallback
    basic_keys = ['mean', 'std', 'min', 'max', 'median', 'rms', 'iqr',
                  'skew', 'kurtosis', 'zcr', 'peak_amp', 'range', 'energy',
                  'p25', 'p75']
    freq_keys = ['dom_freq', 'spectral_energy', 'spectral_entropy',
                 'band_vlf', 'band_lf', 'band_hf', 'spectral_centroid']
    nk_keys = ['signal_quality_mean', 'signal_quality_std',
               'n_peaks', 'peak_rate',
               'ibi_mean', 'ibi_std', 'ibi_median', 'ibi_rmssd', 'ibi_sdnn',
               'ibi_cvsd', 'ibi_range', 'ibi_iqr',
               'hrv_mean_hr', 'hrv_std_hr',
               'peak_amp_mean', 'peak_amp_std', 'peak_amp_range',
               'hrv_lf', 'hrv_hf', 'hrv_lf_hf_ratio', 'hrv_vlf',
               'hrv_pnn50', 'hrv_pnn20',
               'hrv_sd1', 'hrv_sd2', 'hrv_sd_ratio',
               'hrv_sample_entropy']

    all_keys = basic_keys + freq_keys + nk_keys
    nan_feats = {f'ppg_{k}': np.nan for k in all_keys}

    if window is None:
        return nan_feats

    ppg_raw = window['PPG Green'].values.astype(float)
    t_diff = window['T'].diff().dt.total_seconds().dropna()
    sr = 1.0 / t_diff.median() if len(t_diff) > 0 and t_diff.median() > 0 else 25.0
    sr = int(round(sr))
    if sr < 1:
        sr = 25

    feats = {}

    # --- Basic time-domain features on raw PPG ---
    feats['ppg_mean'] = safe_stat(np.mean, ppg_raw)
    feats['ppg_std'] = safe_stat(np.std, ppg_raw)
    feats['ppg_min'] = safe_stat(np.min, ppg_raw)
    feats['ppg_max'] = safe_stat(np.max, ppg_raw)
    feats['ppg_median'] = safe_stat(np.median, ppg_raw)
    feats['ppg_rms'] = safe_stat(lambda a: np.sqrt(np.mean(a**2)), ppg_raw)
    feats['ppg_iqr'] = np.percentile(ppg_raw, 75) - np.percentile(ppg_raw, 25)
    feats['ppg_skew'] = safe_stat(stats.skew, ppg_raw)
    feats['ppg_kurtosis'] = safe_stat(stats.kurtosis, ppg_raw)
    feats['ppg_range'] = np.max(ppg_raw) - np.min(ppg_raw)
    feats['ppg_peak_amp'] = np.max(np.abs(ppg_raw))
    feats['ppg_energy'] = safe_stat(lambda a: np.sum(a**2) / len(a), ppg_raw)
    feats['ppg_p25'] = np.percentile(ppg_raw, 25)
    feats['ppg_p75'] = np.percentile(ppg_raw, 75)
    centered = ppg_raw - np.mean(ppg_raw)
    feats['ppg_zcr'] = np.sum(np.diff(np.sign(centered)) != 0) / len(ppg_raw)

    # --- Frequency-domain features on raw PPG ---
    if len(ppg_raw) >= 8:
        n = len(ppg_raw)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        fft_vals = np.abs(np.fft.rfft(ppg_raw - np.mean(ppg_raw)))
        power = fft_vals ** 2
        total_power = np.sum(power) + 1e-10

        feats['ppg_dom_freq'] = freqs[np.argmax(power[1:]) + 1] if len(power) > 1 else 0.0
        feats['ppg_spectral_energy'] = np.sum(power)
        psd_norm = power / total_power
        psd_norm = psd_norm[psd_norm > 0]
        feats['ppg_spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm))
        feats['ppg_spectral_centroid'] = np.sum(freqs * power) / total_power

        # PPG-specific bands: VLF (0-0.04Hz), LF (0.04-0.15Hz), HF (0.15-0.4Hz)
        vlf_mask = freqs <= 0.04
        lf_mask = (freqs > 0.04) & (freqs <= 0.15)
        hf_mask = (freqs > 0.15) & (freqs <= 0.4)
        feats['ppg_band_vlf'] = np.sum(power[vlf_mask]) / total_power if np.any(vlf_mask) else 0.0
        feats['ppg_band_lf'] = np.sum(power[lf_mask]) / total_power if np.any(lf_mask) else 0.0
        feats['ppg_band_hf'] = np.sum(power[hf_mask]) / total_power if np.any(hf_mask) else 0.0
    else:
        for k in freq_keys:
            feats[f'ppg_{k}'] = np.nan

    # --- NeuroKit2-based features ---
    try:
        # Clean and process PPG signal
        ppg_cleaned = nk.ppg_clean(ppg_raw, sampling_rate=sr)

        # Signal quality
        try:
            quality = nk.ppg_quality(ppg_cleaned, sampling_rate=sr)
            if isinstance(quality, (np.ndarray, pd.Series)):
                feats['ppg_signal_quality_mean'] = np.nanmean(quality)
                feats['ppg_signal_quality_std'] = np.nanstd(quality)
            else:
                feats['ppg_signal_quality_mean'] = float(quality)
                feats['ppg_signal_quality_std'] = 0.0
        except Exception:
            feats['ppg_signal_quality_mean'] = np.nan
            feats['ppg_signal_quality_std'] = np.nan

        # Peak detection
        try:
            peaks_info = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=sr)
            peak_locs = peaks_info.get('PPG_Peaks', [])
            if isinstance(peak_locs, np.ndarray):
                peak_locs = peak_locs.tolist()

            n_peaks = len(peak_locs)
            feats['ppg_n_peaks'] = n_peaks
            duration_sec = len(ppg_cleaned) / sr
            feats['ppg_peak_rate'] = n_peaks / duration_sec if duration_sec > 0 else 0.0

            if n_peaks >= 2:
                # Inter-beat intervals
                ibi = np.diff(peak_locs) / sr  # in seconds
                feats['ppg_ibi_mean'] = np.mean(ibi)
                feats['ppg_ibi_std'] = np.std(ibi)
                feats['ppg_ibi_median'] = np.median(ibi)
                feats['ppg_ibi_range'] = np.max(ibi) - np.min(ibi)
                feats['ppg_ibi_iqr'] = np.percentile(ibi, 75) - np.percentile(ibi, 25)

                # RMSSD
                successive_diffs = np.diff(ibi)
                feats['ppg_ibi_rmssd'] = np.sqrt(np.mean(successive_diffs**2)) if len(successive_diffs) > 0 else np.nan
                # SDNN
                feats['ppg_ibi_sdnn'] = np.std(ibi, ddof=1) if len(ibi) > 1 else np.nan
                # CVSD
                feats['ppg_ibi_cvsd'] = feats['ppg_ibi_rmssd'] / feats['ppg_ibi_mean'] if feats['ppg_ibi_mean'] > 0 else np.nan

                # Heart rate stats
                hr = 60.0 / ibi
                feats['ppg_hrv_mean_hr'] = np.mean(hr)
                feats['ppg_hrv_std_hr'] = np.std(hr)

                # Peak amplitudes
                valid_peaks = [p for p in peak_locs if p < len(ppg_cleaned)]
                if valid_peaks:
                    peak_amps = ppg_cleaned[valid_peaks]
                    feats['ppg_peak_amp_mean'] = np.mean(peak_amps)
                    feats['ppg_peak_amp_std'] = np.std(peak_amps)
                    feats['ppg_peak_amp_range'] = np.max(peak_amps) - np.min(peak_amps)
                else:
                    feats['ppg_peak_amp_mean'] = np.nan
                    feats['ppg_peak_amp_std'] = np.nan
                    feats['ppg_peak_amp_range'] = np.nan

                # pNN50 and pNN20
                if len(successive_diffs) > 0:
                    abs_diffs = np.abs(successive_diffs)
                    feats['ppg_hrv_pnn50'] = np.sum(abs_diffs > 0.05) / len(abs_diffs)
                    feats['ppg_hrv_pnn20'] = np.sum(abs_diffs > 0.02) / len(abs_diffs)
                else:
                    feats['ppg_hrv_pnn50'] = np.nan
                    feats['ppg_hrv_pnn20'] = np.nan

                # Poincare plot features (SD1, SD2)
                if len(ibi) >= 3:
                    ibi1 = ibi[:-1]
                    ibi2 = ibi[1:]
                    sd1 = np.std(ibi2 - ibi1) / np.sqrt(2)
                    sd2 = np.std(ibi2 + ibi1) / np.sqrt(2)
                    feats['ppg_hrv_sd1'] = sd1
                    feats['ppg_hrv_sd2'] = sd2
                    feats['ppg_hrv_sd_ratio'] = sd1 / sd2 if sd2 > 0 else np.nan
                else:
                    feats['ppg_hrv_sd1'] = np.nan
                    feats['ppg_hrv_sd2'] = np.nan
                    feats['ppg_hrv_sd_ratio'] = np.nan

                # HRV frequency-domain from IBI
                if len(ibi) >= 4:
                    try:
                        # Interpolate IBI to uniform sampling for FFT
                        ibi_times = np.cumsum(ibi)
                        ibi_sr = 4  # 4 Hz interpolation
                        t_uniform = np.arange(0, ibi_times[-1], 1.0 / ibi_sr)
                        if len(t_uniform) >= 4:
                            ibi_interp = np.interp(t_uniform, ibi_times, ibi)
                            ibi_centered = ibi_interp - np.mean(ibi_interp)
                            n_ibi = len(ibi_centered)
                            freqs_ibi = np.fft.rfftfreq(n_ibi, d=1.0 / ibi_sr)
                            fft_ibi = np.abs(np.fft.rfft(ibi_centered)) ** 2
                            total_p = np.sum(fft_ibi) + 1e-10

                            vlf_m = freqs_ibi <= 0.04
                            lf_m = (freqs_ibi > 0.04) & (freqs_ibi <= 0.15)
                            hf_m = (freqs_ibi > 0.15) & (freqs_ibi <= 0.4)

                            feats['ppg_hrv_vlf'] = np.sum(fft_ibi[vlf_m]) / total_p if np.any(vlf_m) else 0.0
                            feats['ppg_hrv_lf'] = np.sum(fft_ibi[lf_m]) / total_p if np.any(lf_m) else 0.0
                            feats['ppg_hrv_hf'] = np.sum(fft_ibi[hf_m]) / total_p if np.any(hf_m) else 0.0
                            feats['ppg_hrv_lf_hf_ratio'] = feats['ppg_hrv_lf'] / feats['ppg_hrv_hf'] if feats['ppg_hrv_hf'] > 0 else np.nan
                        else:
                            for k in ['hrv_vlf', 'hrv_lf', 'hrv_hf', 'hrv_lf_hf_ratio']:
                                feats[f'ppg_{k}'] = np.nan
                    except Exception:
                        for k in ['hrv_vlf', 'hrv_lf', 'hrv_hf', 'hrv_lf_hf_ratio']:
                            feats[f'ppg_{k}'] = np.nan
                else:
                    for k in ['hrv_vlf', 'hrv_lf', 'hrv_hf', 'hrv_lf_hf_ratio']:
                        feats[f'ppg_{k}'] = np.nan

                # Sample entropy of IBI
                try:
                    feats['ppg_hrv_sample_entropy'] = nk.entropy_sample(ibi, dimension=2, tolerance=0.2 * np.std(ibi))[0]
                except Exception:
                    feats['ppg_hrv_sample_entropy'] = np.nan

            else:
                # Not enough peaks for IBI-based features
                for k in nk_keys:
                    if k not in ['signal_quality_mean', 'signal_quality_std', 'n_peaks', 'peak_rate']:
                        feats[f'ppg_{k}'] = feats.get(f'ppg_{k}', np.nan)

        except Exception:
            for k in nk_keys:
                if k not in ['signal_quality_mean', 'signal_quality_std']:
                    feats[f'ppg_{k}'] = np.nan

    except Exception:
        # NeuroKit processing failed entirely — fill NaN for all nk_keys
        for k in nk_keys:
            feats[f'ppg_{k}'] = np.nan

    # Fill any missing keys
    for k in all_keys:
        if f'ppg_{k}' not in feats:
            feats[f'ppg_{k}'] = np.nan

    return feats


# ===================== LIGHT FEATURES =====================

def extract_light_features(window):
    """Extract time-domain and trend features from light sensor."""
    all_keys = ['mean', 'std', 'min', 'max', 'median', 'rms', 'iqr',
                'skew', 'kurtosis', 'zcr', 'peak_amp', 'range', 'energy',
                'p25', 'p75', 'slope', 'log_mean', 'log_std',
                'change_rate_mean', 'change_rate_std', 'n_changes']
    nan_feats = {f'light_{k}': np.nan for k in all_keys}

    if window is None:
        return nan_feats

    values = window['Light'].values.astype(float)
    feats = {}

    # Time-domain
    feats['light_mean'] = safe_stat(np.mean, values)
    feats['light_std'] = safe_stat(np.std, values)
    feats['light_min'] = safe_stat(np.min, values)
    feats['light_max'] = safe_stat(np.max, values)
    feats['light_median'] = safe_stat(np.median, values)
    feats['light_rms'] = safe_stat(lambda a: np.sqrt(np.mean(a**2)), values)
    feats['light_iqr'] = np.percentile(values, 75) - np.percentile(values, 25)
    feats['light_skew'] = safe_stat(stats.skew, values)
    feats['light_kurtosis'] = safe_stat(stats.kurtosis, values)
    feats['light_range'] = np.max(values) - np.min(values)
    feats['light_peak_amp'] = np.max(np.abs(values))
    feats['light_energy'] = safe_stat(lambda a: np.sum(a**2) / len(a), values)
    feats['light_p25'] = np.percentile(values, 25)
    feats['light_p75'] = np.percentile(values, 75)

    centered = values - np.mean(values)
    feats['light_zcr'] = np.sum(np.diff(np.sign(centered)) != 0) / len(values)

    # Slope / trend
    if len(values) >= 3:
        t = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(t, values)
        feats['light_slope'] = slope
    else:
        feats['light_slope'] = np.nan

    # Log-domain (light can span orders of magnitude)
    log_vals = np.log1p(np.clip(values, 0, None))
    feats['light_log_mean'] = safe_stat(np.mean, log_vals)
    feats['light_log_std'] = safe_stat(np.std, log_vals)

    # Rate of change
    if len(values) >= 3:
        diffs = np.diff(values)
        feats['light_change_rate_mean'] = np.mean(np.abs(diffs))
        feats['light_change_rate_std'] = np.std(diffs)
        # Number of significant changes (> 10% of range)
        rng = np.max(values) - np.min(values)
        threshold = rng * 0.1 if rng > 0 else 1.0
        feats['light_n_changes'] = np.sum(np.abs(diffs) > threshold)
    else:
        feats['light_change_rate_mean'] = np.nan
        feats['light_change_rate_std'] = np.nan
        feats['light_n_changes'] = np.nan

    return feats


# ===================== MAIN PIPELINE =====================

def extract_all_features(acc_window, ppg_window, light_window):
    feats = {}
    feats.update(extract_acc_features(acc_window))
    feats.update(extract_ppg_features(ppg_window))
    feats.update(extract_light_features(light_window))
    return feats


def process_all_subjects():
    labels = load_labels()
    subjects = sorted(labels['P_ID'].unique())

    print(f"Processing {len(labels)} samples across {len(subjects)} subjects")
    print(f"Window: first {WINDOW_SECONDS} seconds of each 90s segment")
    print(f"Output: {OUTPUT_DIR}")
    print()

    all_features = []
    missing_counts = {'acc': 0, 'ppg': 0, 'light': 0}

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
        n_samples = len(subject_labels)
        print(f"  [{i+1:2d}/{len(subjects)}] {subject}: {n_samples:>5d} samples, {elapsed:6.1f}s")

    # Save single combined CSV
    df = pd.DataFrame(all_features)
    out_path = OUTPUT_DIR / 'all_subjects_features.csv'
    df.to_csv(out_path, index=False)

    feat_cols = [c for c in df.columns if c not in ['P_ID', 'category', 'Sensor_start_time']]
    nan_rows = df[feat_cols].isna().any(axis=1).sum()

    print(f"\n{'='*50}")
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(feat_cols)}")
    print(f"Missing windows — ACC: {missing_counts['acc']}, PPG: {missing_counts['ppg']}, Light: {missing_counts['light']}")
    print(f"Rows with any NaN: {nan_rows}")
    print(f"Class distribution: {dict(df['category'].value_counts().sort_index())}")
    print(f"\nSaved to: {out_path}")

    return df


if __name__ == '__main__':
    df = process_all_subjects()

    # Print feature summary
    feat_cols = [c for c in df.columns if c not in ['P_ID', 'category', 'Sensor_start_time']]
    acc_feats = [c for c in feat_cols if c.startswith('acc_')]
    ppg_feats = [c for c in feat_cols if c.startswith('ppg_')]
    light_feats = [c for c in feat_cols if c.startswith('light_')]
    print(f"\nFeature breakdown:")
    print(f"  Accelerometer: {len(acc_feats)} features")
    print(f"  PPG:           {len(ppg_feats)} features")
    print(f"  Light:         {len(light_feats)} features")
    print(f"  Total:         {len(feat_cols)} features")
