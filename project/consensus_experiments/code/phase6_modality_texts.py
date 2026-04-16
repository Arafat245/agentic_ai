"""
Phase 6a: Generate per-modality text descriptions for ConSensus-style multi-agent experiments.
Splits the combined text template into separate ACC, PPG, and Light descriptions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path('/mnt/sdb/arafat/agentic_ai/project')
FEATURES_FILE = PROJECT_DIR / 'temp_exps' / 'all_subjects_features.csv'
OUTPUT_FILE = PROJECT_DIR / 'temp_exps' / 'modality_text_features.csv'


def fmt(val, decimals=2):
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return "unavailable"
    return f"{val:.{decimals}f}"


def build_acc_text(row):
    return (
        "The following describes accelerometer measurements from a 16-second smartwatch wrist recording.\n"
        "\n"
        f"High-frequency power fractions: x-axis {fmt(row['acc_x_band_high'], 3)}, "
        f"z-axis {fmt(row['acc_z_band_high'], 3)}, y-axis {fmt(row['acc_y_band_high'], 3)}, "
        f"magnitude {fmt(row['acc_mag_band_high'], 3)}. "
        f"Low-frequency power fractions: x-axis {fmt(row['acc_x_band_low'], 3)}, "
        f"z-axis {fmt(row['acc_z_band_low'], 3)}, y-axis {fmt(row['acc_y_band_low'], 3)}.\n"
        "\n"
        f"Spectral centroids: x-axis {fmt(row['acc_x_spectral_centroid'])} Hz, "
        f"z-axis {fmt(row['acc_z_spectral_centroid'])} Hz, "
        f"y-axis {fmt(row['acc_y_spectral_centroid'])} Hz. "
        f"Spectral entropy: x-axis {fmt(row['acc_x_spectral_entropy'])}, "
        f"z-axis {fmt(row['acc_z_spectral_entropy'])}.\n"
        "\n"
        f"Zero-crossing rates: x-axis {fmt(row['acc_x_zcr'], 3)}, "
        f"z-axis {fmt(row['acc_z_zcr'], 3)}, y-axis {fmt(row['acc_y_zcr'], 3)}.\n"
        "\n"
        f"Minimums: y-axis {fmt(row['acc_y_min'])}, z-axis {fmt(row['acc_z_min'])}, "
        f"x-axis {fmt(row['acc_x_min'])}, magnitude {fmt(row['acc_mag_min'])}. "
        f"Z-axis range {fmt(row['acc_z_range'])}, std {fmt(row['acc_z_std'])}. "
        f"Magnitude range {fmt(row['acc_mag_range'])}. "
        f"25th percentiles: x-axis {fmt(row['acc_x_p25'])}, y-axis {fmt(row['acc_y_p25'])}. "
        f"Y-axis peak amplitude {fmt(row['acc_y_peak_amp'])}. "
        f"X-axis dominant frequency {fmt(row['acc_x_dom_freq'])} Hz."
    )


def build_ppg_text(row):
    return (
        "The following describes PPG (photoplethysmography) measurements from a 16-second smartwatch wrist recording.\n"
        "\n"
        f"Signal quality: mean {fmt(row['ppg_signal_quality_mean'], 3)}, "
        f"std {fmt(row['ppg_signal_quality_std'], 3)}. "
        f"Heart rate variability: pNN50 {fmt(row['ppg_hrv_pnn50'], 3)}, "
        f"pNN20 {fmt(row['ppg_hrv_pnn20'], 3)}. "
        f"PPG zero-crossing rate {fmt(row['ppg_zcr'], 4)}, "
        f"peak rate {fmt(row['ppg_peak_rate'])} peaks/sec."
    )


def build_light_text(row):
    return (
        "The following describes ambient light measurements from a 16-second smartwatch recording.\n"
        "\n"
        f"Log-transformed light level: mean {fmt(row['light_log_mean'], 3)}, "
        f"std {fmt(row['light_log_std'], 3)}. "
        f"Distribution kurtosis {fmt(row['light_kurtosis'])}. "
        f"Number of significant light changes: {fmt(row['light_n_changes'], 0)}."
    )


def build_all_text(row):
    """Combined text for single-agent baseline (same as phase3 but regenerated for consistency)."""
    return (
        "The following describes a 16-second smartwatch sensor recording from a participant's wrist.\n"
        "\n"
        f"Accelerometer: High-freq power x={fmt(row['acc_x_band_high'], 3)}, "
        f"z={fmt(row['acc_z_band_high'], 3)}, y={fmt(row['acc_y_band_high'], 3)}, "
        f"mag={fmt(row['acc_mag_band_high'], 3)}. "
        f"Low-freq power x={fmt(row['acc_x_band_low'], 3)}, z={fmt(row['acc_z_band_low'], 3)}, "
        f"y={fmt(row['acc_y_band_low'], 3)}. "
        f"Spectral centroids x={fmt(row['acc_x_spectral_centroid'])} Hz, "
        f"z={fmt(row['acc_z_spectral_centroid'])} Hz, y={fmt(row['acc_y_spectral_centroid'])} Hz. "
        f"Spectral entropy x={fmt(row['acc_x_spectral_entropy'])}, z={fmt(row['acc_z_spectral_entropy'])}. "
        f"ZCR x={fmt(row['acc_x_zcr'], 3)}, z={fmt(row['acc_z_zcr'], 3)}, y={fmt(row['acc_y_zcr'], 3)}. "
        f"Min y={fmt(row['acc_y_min'])}, z={fmt(row['acc_z_min'])}, x={fmt(row['acc_x_min'])}, "
        f"mag={fmt(row['acc_mag_min'])}. "
        f"Z-range={fmt(row['acc_z_range'])}, z-std={fmt(row['acc_z_std'])}, "
        f"mag-range={fmt(row['acc_mag_range'])}. "
        f"P25 x={fmt(row['acc_x_p25'])}, y={fmt(row['acc_y_p25'])}. "
        f"Y-peak-amp={fmt(row['acc_y_peak_amp'])}, x-dom-freq={fmt(row['acc_x_dom_freq'])} Hz.\n"
        "\n"
        f"PPG: Signal quality mean={fmt(row['ppg_signal_quality_mean'], 3)}, "
        f"std={fmt(row['ppg_signal_quality_std'], 3)}. "
        f"HRV pNN50={fmt(row['ppg_hrv_pnn50'], 3)}, pNN20={fmt(row['ppg_hrv_pnn20'], 3)}. "
        f"ZCR={fmt(row['ppg_zcr'], 4)}, peak rate={fmt(row['ppg_peak_rate'])} peaks/sec.\n"
        "\n"
        f"Light: Log-mean={fmt(row['light_log_mean'], 3)}, log-std={fmt(row['light_log_std'], 3)}. "
        f"Kurtosis={fmt(row['light_kurtosis'])}, changes={fmt(row['light_n_changes'], 0)}."
    )


def main():
    print("Loading features...")
    df = pd.read_csv(FEATURES_FILE)
    df['category'] = df['category'].astype(int)

    # Fill NaN for the top-30 features used in text templates
    top_features = [
        'acc_x_band_high', 'acc_x_band_low', 'acc_z_band_high',
        'acc_x_spectral_centroid', 'acc_x_zcr', 'acc_y_min',
        'acc_y_band_high', 'acc_mag_band_high', 'acc_z_band_low',
        'acc_z_zcr', 'acc_z_spectral_centroid', 'acc_z_min',
        'acc_z_range', 'acc_x_spectral_entropy', 'acc_z_std',
        'acc_x_min', 'acc_y_spectral_centroid', 'acc_mag_min',
        'acc_y_zcr', 'acc_mag_range',
        'ppg_signal_quality_mean', 'ppg_hrv_pnn50',
        'ppg_signal_quality_std', 'ppg_hrv_pnn20',
        'ppg_zcr', 'ppg_peak_rate',
        'light_log_mean', 'light_kurtosis', 'light_log_std', 'light_n_changes',
    ]
    df[top_features] = df[top_features].fillna(0)

    print(f"Loaded {len(df)} samples, {df['P_ID'].nunique()} subjects")
    print("Generating per-modality text descriptions...")

    df['text_acc'] = df.apply(build_acc_text, axis=1)
    df['text_ppg'] = df.apply(build_ppg_text, axis=1)
    df['text_light'] = df.apply(build_light_text, axis=1)
    df['text_all'] = df.apply(build_all_text, axis=1)

    out_df = df[['P_ID', 'category', 'Sensor_start_time', 'text_acc', 'text_ppg', 'text_light', 'text_all']]
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")
    print(f"Rows: {len(out_df)}")

    # Show sample
    sample = out_df.iloc[0]
    print(f"\n{'='*60}")
    print(f"Sample ACC text ({len(sample['text_acc'].split())} words):")
    print(sample['text_acc'][:300])
    print(f"\nSample PPG text ({len(sample['text_ppg'].split())} words):")
    print(sample['text_ppg'])
    print(f"\nSample Light text ({len(sample['text_light'].split())} words):")
    print(sample['text_light'])
    print(f"\nSample ALL text ({len(sample['text_all'].split())} words):")
    print(sample['text_all'][:300])


if __name__ == '__main__':
    main()
