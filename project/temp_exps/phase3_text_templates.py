"""
Phase 3: Convert numeric sensor features into natural language text templates.
Follows SensorLM's statistical caption approach — plug feature values into fixed templates.
Uses ALL 164 features in full natural language sentences.
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
    """Convert a row of all 164 features into natural language prose (SensorLM style)."""

    text = (
        f"The following describes a 16-second smartwatch sensor recording from a participant's wrist.\n"
        f"\n"
        # ============ ACCELEROMETER X-AXIS ============
        f"The x-axis accelerometer readings show a mean of {fmt(row['acc_x_mean'])} with a standard deviation of {fmt(row['acc_x_std'])}. "
        f"The values range from a minimum of {fmt(row['acc_x_min'])} to a maximum of {fmt(row['acc_x_max'])}, "
        f"giving a total range of {fmt(row['acc_x_range'])}. "
        f"The median x-axis acceleration is {fmt(row['acc_x_median'])}, with the 25th percentile at {fmt(row['acc_x_p25'])} "
        f"and the 75th percentile at {fmt(row['acc_x_p75'])}, yielding an interquartile range of {fmt(row['acc_x_iqr'])}. "
        f"The root mean square value is {fmt(row['acc_x_rms'])} and the peak amplitude reaches {fmt(row['acc_x_peak_amp'])}. "
        f"The signal energy is {fmt(row['acc_x_energy'])}. "
        f"The distribution has a skewness of {fmt(row['acc_x_skew'])} and kurtosis of {fmt(row['acc_x_kurtosis'])}. "
        f"The zero-crossing rate is {fmt(row['acc_x_zcr'], 3)}. "
        f"In the frequency domain, the dominant frequency is {fmt(row['acc_x_dom_freq'])} Hz "
        f"with a spectral energy of {fmt(row['acc_x_spectral_energy'])} and spectral entropy of {fmt(row['acc_x_spectral_entropy'])}. "
        f"The spectral centroid is {fmt(row['acc_x_spectral_centroid'])} Hz. "
        f"The power distribution across frequency bands is {fmt(row['acc_x_band_low'], 3)} in the low band, "
        f"{fmt(row['acc_x_band_mid'], 3)} in the mid band, and {fmt(row['acc_x_band_high'], 3)} in the high band.\n"
        f"\n"
        # ============ ACCELEROMETER Y-AXIS ============
        f"The y-axis accelerometer readings show a mean of {fmt(row['acc_y_mean'])} with a standard deviation of {fmt(row['acc_y_std'])}. "
        f"The values range from a minimum of {fmt(row['acc_y_min'])} to a maximum of {fmt(row['acc_y_max'])}, "
        f"giving a total range of {fmt(row['acc_y_range'])}. "
        f"The median is {fmt(row['acc_y_median'])}, with percentiles at {fmt(row['acc_y_p25'])} (25th) "
        f"and {fmt(row['acc_y_p75'])} (75th), and an interquartile range of {fmt(row['acc_y_iqr'])}. "
        f"The root mean square is {fmt(row['acc_y_rms'])} and peak amplitude is {fmt(row['acc_y_peak_amp'])}. "
        f"The signal energy is {fmt(row['acc_y_energy'])}. "
        f"The skewness is {fmt(row['acc_y_skew'])} and kurtosis is {fmt(row['acc_y_kurtosis'])}. "
        f"The zero-crossing rate is {fmt(row['acc_y_zcr'], 3)}. "
        f"The dominant frequency is {fmt(row['acc_y_dom_freq'])} Hz with spectral energy of {fmt(row['acc_y_spectral_energy'])} "
        f"and spectral entropy of {fmt(row['acc_y_spectral_entropy'])}. "
        f"The spectral centroid is {fmt(row['acc_y_spectral_centroid'])} Hz. "
        f"The frequency band powers are {fmt(row['acc_y_band_low'], 3)} (low), "
        f"{fmt(row['acc_y_band_mid'], 3)} (mid), and {fmt(row['acc_y_band_high'], 3)} (high).\n"
        f"\n"
        # ============ ACCELEROMETER Z-AXIS ============
        f"The z-axis accelerometer readings show a mean of {fmt(row['acc_z_mean'])} with a standard deviation of {fmt(row['acc_z_std'])}. "
        f"The values range from {fmt(row['acc_z_min'])} to {fmt(row['acc_z_max'])}, "
        f"with a range of {fmt(row['acc_z_range'])}. "
        f"The median is {fmt(row['acc_z_median'])}, with percentiles at {fmt(row['acc_z_p25'])} (25th) "
        f"and {fmt(row['acc_z_p75'])} (75th), and an interquartile range of {fmt(row['acc_z_iqr'])}. "
        f"The root mean square is {fmt(row['acc_z_rms'])} and peak amplitude is {fmt(row['acc_z_peak_amp'])}. "
        f"The signal energy is {fmt(row['acc_z_energy'])}. "
        f"The skewness is {fmt(row['acc_z_skew'])} and kurtosis is {fmt(row['acc_z_kurtosis'])}. "
        f"The zero-crossing rate is {fmt(row['acc_z_zcr'], 3)}. "
        f"The dominant frequency is {fmt(row['acc_z_dom_freq'])} Hz with spectral energy of {fmt(row['acc_z_spectral_energy'])} "
        f"and spectral entropy of {fmt(row['acc_z_spectral_entropy'])}. "
        f"The spectral centroid is {fmt(row['acc_z_spectral_centroid'])} Hz. "
        f"The frequency band powers are {fmt(row['acc_z_band_low'], 3)} (low), "
        f"{fmt(row['acc_z_band_mid'], 3)} (mid), and {fmt(row['acc_z_band_high'], 3)} (high).\n"
        f"\n"
        # ============ ACCELEROMETER MAGNITUDE ============
        f"The overall acceleration magnitude has a mean of {fmt(row['acc_mag_mean'])} with a standard deviation of {fmt(row['acc_mag_std'])}. "
        f"The magnitude ranges from {fmt(row['acc_mag_min'])} to {fmt(row['acc_mag_max'])}, "
        f"with a total range of {fmt(row['acc_mag_range'])}. "
        f"The median magnitude is {fmt(row['acc_mag_median'])}, with percentiles at {fmt(row['acc_mag_p25'])} (25th) "
        f"and {fmt(row['acc_mag_p75'])} (75th), and an interquartile range of {fmt(row['acc_mag_iqr'])}. "
        f"The root mean square is {fmt(row['acc_mag_rms'])} and peak amplitude is {fmt(row['acc_mag_peak_amp'])}. "
        f"The signal energy is {fmt(row['acc_mag_energy'])}. "
        f"The skewness is {fmt(row['acc_mag_skew'])} and kurtosis is {fmt(row['acc_mag_kurtosis'])}. "
        f"The zero-crossing rate is {fmt(row['acc_mag_zcr'], 3)}. "
        f"The dominant frequency is {fmt(row['acc_mag_dom_freq'])} Hz with spectral energy of {fmt(row['acc_mag_spectral_energy'])} "
        f"and spectral entropy of {fmt(row['acc_mag_spectral_entropy'])}. "
        f"The spectral centroid is {fmt(row['acc_mag_spectral_centroid'])} Hz. "
        f"The frequency band powers are {fmt(row['acc_mag_band_low'], 3)} (low), "
        f"{fmt(row['acc_mag_band_mid'], 3)} (mid), and {fmt(row['acc_mag_band_high'], 3)} (high).\n"
        f"\n"
        # ============ ACCELEROMETER CROSS-AXIS & DERIVED ============
        f"The cross-axis correlations are {fmt(row['acc_corr_xy'])} between x and y, "
        f"{fmt(row['acc_corr_xz'])} between x and z, and {fmt(row['acc_corr_yz'])} between y and z. "
        f"The signal magnitude area is {fmt(row['acc_sma'])}. "
        f"The average jerk of the movement is {fmt(row['acc_jerk_mean'])} with a standard deviation of {fmt(row['acc_jerk_std'])}.\n"
        f"\n"
        # ============ PPG BASIC STATS ============
        f"The PPG green channel signal has a mean of {fmt(row['ppg_mean'], 0)} with a standard deviation of {fmt(row['ppg_std'], 0)}. "
        f"The values range from {fmt(row['ppg_min'], 0)} to {fmt(row['ppg_max'], 0)}, "
        f"giving a range of {fmt(row['ppg_range'], 0)}. "
        f"The median is {fmt(row['ppg_median'], 0)}, with percentiles at {fmt(row['ppg_p25'], 0)} (25th) "
        f"and {fmt(row['ppg_p75'], 0)} (75th), and an interquartile range of {fmt(row['ppg_iqr'], 0)}. "
        f"The root mean square is {fmt(row['ppg_rms'], 0)} and peak amplitude is {fmt(row['ppg_peak_amp'], 0)}. "
        f"The signal energy is {fmt(row['ppg_energy'], 0)}. "
        f"The skewness is {fmt(row['ppg_skew'])} and kurtosis is {fmt(row['ppg_kurtosis'])}. "
        f"The zero-crossing rate is {fmt(row['ppg_zcr'], 4)}.\n"
        f"\n"
        # ============ PPG FREQUENCY DOMAIN ============
        f"In the frequency domain, the dominant PPG frequency is {fmt(row['ppg_dom_freq'], 3)} Hz "
        f"with spectral energy of {fmt(row['ppg_spectral_energy'], 0)} and spectral entropy of {fmt(row['ppg_spectral_entropy'])}. "
        f"The spectral centroid is {fmt(row['ppg_spectral_centroid'], 3)} Hz. "
        f"The PPG frequency band powers are {fmt(row['ppg_band_vlf'], 4)} (VLF), "
        f"{fmt(row['ppg_band_lf'], 4)} (LF), and {fmt(row['ppg_band_hf'], 4)} (HF).\n"
        f"\n"
        # ============ PPG NEUROKIT2 FEATURES ============
        f"The PPG signal quality averages {fmt(row['ppg_signal_quality_mean'], 3)} "
        f"with a standard deviation of {fmt(row['ppg_signal_quality_std'], 3)}. "
        f"The signal contains {fmt(row['ppg_n_peaks'], 0)} detected peaks at a rate of {fmt(row['ppg_peak_rate'])} peaks per second. "
        f"The mean peak amplitude is {fmt(row['ppg_peak_amp_mean'], 0)} with a standard deviation of {fmt(row['ppg_peak_amp_std'], 0)} "
        f"and a range of {fmt(row['ppg_peak_amp_range'], 0)}.\n"
        f"\n"
        # ============ PPG HRV / IBI ============
        f"The estimated heart rate is {fmt(row['ppg_hrv_mean_hr'], 1)} beats per minute "
        f"with a standard deviation of {fmt(row['ppg_hrv_std_hr'], 1)} bpm. "
        f"The mean inter-beat interval is {fmt(row['ppg_ibi_mean'], 3)} seconds "
        f"with a standard deviation of {fmt(row['ppg_ibi_std'], 3)} seconds. "
        f"The median inter-beat interval is {fmt(row['ppg_ibi_median'], 3)} seconds. "
        f"The inter-beat interval RMSSD is {fmt(row['ppg_ibi_rmssd'], 4)}, SDNN is {fmt(row['ppg_ibi_sdnn'], 4)}, "
        f"and CVSD is {fmt(row['ppg_ibi_cvsd'], 4)}. "
        f"The inter-beat interval range is {fmt(row['ppg_ibi_range'], 3)} seconds "
        f"and the interquartile range is {fmt(row['ppg_ibi_iqr'], 3)} seconds. "
        f"The parasympathetic activity markers are pNN50 of {fmt(row['ppg_hrv_pnn50'], 3)} "
        f"and pNN20 of {fmt(row['ppg_hrv_pnn20'], 3)}.\n"
        f"\n"
        # ============ PPG HRV ADVANCED ============
        f"The Poincare plot analysis shows SD1 of {fmt(row['ppg_hrv_sd1'], 4)} and SD2 of {fmt(row['ppg_hrv_sd2'], 4)}, "
        f"with an SD1/SD2 ratio of {fmt(row['ppg_hrv_sd_ratio'], 4)}. "
        f"The HRV frequency domain analysis shows VLF power of {fmt(row['ppg_hrv_vlf'], 4)}, "
        f"LF power of {fmt(row['ppg_hrv_lf'], 4)}, and HF power of {fmt(row['ppg_hrv_hf'], 4)}. "
        f"The LF/HF ratio is {fmt(row['ppg_hrv_lf_hf_ratio'])}. "
        f"The sample entropy of the heart rate variability is {fmt(row['ppg_hrv_sample_entropy'], 4)}.\n"
        f"\n"
        # ============ LIGHT SENSOR ============
        f"The ambient light sensor reads a mean of {fmt(row['light_mean'], 1)} lux "
        f"with a standard deviation of {fmt(row['light_std'], 1)}. "
        f"The light values range from {fmt(row['light_min'], 1)} to {fmt(row['light_max'], 1)}, "
        f"with a total range of {fmt(row['light_range'], 1)}. "
        f"The median light level is {fmt(row['light_median'], 1)}, with percentiles at {fmt(row['light_p25'], 1)} (25th) "
        f"and {fmt(row['light_p75'], 1)} (75th), and an interquartile range of {fmt(row['light_iqr'], 1)}. "
        f"The root mean square is {fmt(row['light_rms'], 1)} and peak amplitude is {fmt(row['light_peak_amp'], 1)}. "
        f"The signal energy is {fmt(row['light_energy'], 1)}. "
        f"The skewness is {fmt(row['light_skew'])} and kurtosis is {fmt(row['light_kurtosis'])}. "
        f"The zero-crossing rate is {fmt(row['light_zcr'], 4)}. "
        f"The light trend slope is {fmt(row['light_slope'], 4)}. "
        f"The log-transformed light level averages {fmt(row['light_log_mean'], 3)} "
        f"with a log standard deviation of {fmt(row['light_log_std'], 3)}. "
        f"The mean rate of light change is {fmt(row['light_change_rate_mean'], 1)} "
        f"with a standard deviation of {fmt(row['light_change_rate_std'], 1)}, "
        f"and the number of significant light changes is {fmt(row['light_n_changes'], 0)}."
    )
    return text


def main():
    print("Loading features...")
    df = pd.read_csv(FEATURES_FILE)
    print(f"Loaded {len(df)} samples, {df['P_ID'].nunique()} subjects")

    print("Generating text descriptions (all 164 features)...")
    df['text_description'] = df.apply(build_text_description, axis=1)

    # Save only the columns needed for LLM experiments
    out_df = df[['P_ID', 'category', 'Sensor_start_time', 'text_description']]
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")
    print(f"Rows: {len(out_df)}")

    # Show an example
    print(f"\n{'='*60}")
    print("Example text description (first sample):")
    print(f"{'='*60}")
    print(out_df.iloc[0]['text_description'])
    print(f"\nCategory: {out_df.iloc[0]['category']}")
    print(f"\nApprox tokens per description: ~{len(out_df.iloc[0]['text_description'].split())}")


if __name__ == '__main__':
    main()
