"""
Phase 3: Convert numeric sensor features into natural language text templates.
Follows SensorLM's statistical caption approach — plug feature values into fixed templates.
Uses top 30 most discriminative features (by Cohen's d) in full natural language sentences.
Output: CSV with P_ID, category, Sensor_start_time, text_description columns.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path('/mnt/sdb/arafat/agentic_ai/project')
FEATURES_FILE = PROJECT_DIR / 'temp_exps' / 'all_subjects_features.csv'
OUTPUT_FILE = PROJECT_DIR / 'temp_exps' / 'text_features.csv'


def fmt(val, decimals=2):
    """Format a value, returning 'unavailable' for NaN/Inf."""
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return "unavailable"
    return f"{val:.{decimals}f}"


def build_text_description(row):
    """Convert a row into natural language prose using the top 30 discriminative features."""

    text = (
        f"The following describes a 16-second smartwatch sensor recording from a participant's wrist.\n"
        f"\n"
        # ACC spectral band powers (ranks 1, 3, 4, 8, 9, 10, 27)
        f"The high-frequency power fraction of the x-axis wrist acceleration is {fmt(row['acc_x_band_high'], 3)}, "
        f"while the low-frequency power fraction is {fmt(row['acc_x_band_low'], 3)}. "
        f"For the z-axis, the high-frequency power is {fmt(row['acc_z_band_high'], 3)} "
        f"and the low-frequency power is {fmt(row['acc_z_band_low'], 3)}. "
        f"The y-axis shows a high-frequency power of {fmt(row['acc_y_band_high'], 3)} "
        f"and low-frequency power of {fmt(row['acc_y_band_low'], 3)}. "
        f"The overall magnitude high-frequency power is {fmt(row['acc_mag_band_high'], 3)}.\n"
        f"\n"
        # ACC spectral centroids and entropy (ranks 5, 12, 16, 21, 29)
        f"The spectral centroid of the x-axis acceleration is {fmt(row['acc_x_spectral_centroid'])} Hz, "
        f"the z-axis spectral centroid is {fmt(row['acc_z_spectral_centroid'])} Hz, "
        f"and the y-axis spectral centroid is {fmt(row['acc_y_spectral_centroid'])} Hz. "
        f"The spectral entropy of the x-axis is {fmt(row['acc_x_spectral_entropy'])} "
        f"and the z-axis spectral entropy is {fmt(row['acc_z_spectral_entropy'])}.\n"
        f"\n"
        # ACC zero-crossing rates (ranks 6, 11, 23)
        f"The zero-crossing rate of the x-axis acceleration is {fmt(row['acc_x_zcr'], 3)}, "
        f"the z-axis zero-crossing rate is {fmt(row['acc_z_zcr'], 3)}, "
        f"and the y-axis zero-crossing rate is {fmt(row['acc_y_zcr'], 3)}.\n"
        f"\n"
        # ACC range, min, std, peak (ranks 7, 14, 15, 18, 20, 22, 24, 25, 26, 28, 30)
        f"The minimum y-axis acceleration is {fmt(row['acc_y_min'])} and the minimum z-axis acceleration is {fmt(row['acc_z_min'])}. "
        f"The minimum x-axis acceleration is {fmt(row['acc_x_min'])} and the minimum overall magnitude is {fmt(row['acc_mag_min'])}. "
        f"The z-axis acceleration range is {fmt(row['acc_z_range'])} with a standard deviation of {fmt(row['acc_z_std'])}. "
        f"The overall magnitude range is {fmt(row['acc_mag_range'])}. "
        f"The 25th percentile of x-axis acceleration is {fmt(row['acc_x_p25'])} "
        f"and the 25th percentile of y-axis acceleration is {fmt(row['acc_y_p25'])}. "
        f"The y-axis peak amplitude is {fmt(row['acc_y_peak_amp'])}. "
        f"The dominant frequency of the x-axis acceleration is {fmt(row['acc_x_dom_freq'])} Hz.\n"
        f"\n"
        # PPG features (ranks 2, 13, 17, 35, 46, 56)
        f"The PPG signal quality averages {fmt(row['ppg_signal_quality_mean'], 3)} "
        f"with a standard deviation of {fmt(row['ppg_signal_quality_std'], 3)}. "
        f"The parasympathetic activity marker pNN50 is {fmt(row['ppg_hrv_pnn50'], 3)} "
        f"and pNN20 is {fmt(row['ppg_hrv_pnn20'], 3)}. "
        f"The PPG zero-crossing rate is {fmt(row['ppg_zcr'], 4)} "
        f"and the PPG peak rate is {fmt(row['ppg_peak_rate'])} peaks per second.\n"
        f"\n"
        # Light features (ranks 19, 57, 65, 74)
        f"The log-transformed ambient light level averages {fmt(row['light_log_mean'], 3)} "
        f"with a log standard deviation of {fmt(row['light_log_std'], 3)}. "
        f"The light distribution kurtosis is {fmt(row['light_kurtosis'])} "
        f"and the number of significant light changes is {fmt(row['light_n_changes'], 0)}."
    )
    return text


def main():
    print("Loading features...")
    df = pd.read_csv(FEATURES_FILE)
    print(f"Loaded {len(df)} samples, {df['P_ID'].nunique()} subjects")

    print("Generating text descriptions (top 30 features)...")
    df['text_description'] = df.apply(build_text_description, axis=1)

    out_df = df[['P_ID', 'category', 'Sensor_start_time', 'text_description']]
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")
    print(f"Rows: {len(out_df)}")

    print(f"\n{'='*60}")
    print("Example text description (first sample):")
    print(f"{'='*60}")
    print(out_df.iloc[0]['text_description'])
    print(f"\nCategory: {out_df.iloc[0]['category']}")
    print(f"Approx words per description: ~{len(out_df.iloc[0]['text_description'].split())}")


if __name__ == '__main__':
    main()
