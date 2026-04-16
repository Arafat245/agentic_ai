# Social Interaction Detection from Smartwatch Sensor Data

## Task
Binary classification: predict whether a social interaction occurred (1) or not (0) from the **first 16 seconds** of a 90-second smartwatch sensor window.

## Data
- **Subjects**: 38 (PA01–PA24, PB01–PB18)
- **Samples**: 33,727 total
- **Sensors**: Accelerometer (X/Y/Z), PPG (Green, processed with NeuroKit2), Light
- **Features**: 164 extracted, **top 30 selected** by Cohen's d effect size
- **Class distribution**: 68.5% no interaction (0), 31.5% interaction (1)
- **Evaluation**: Leave-One-Subject-Out Cross-Validation (LOSO-CV), 38 folds

## Top 30 Most Discriminative Features (Cohen's d)

| Rank | Feature | Mean (No Int.) | Std (No Int.) | Mean (Int.) | Std (Int.) | Cohen's d | p-value |
|------|---------|---------------|--------------|------------|-----------|-----------|---------|
| 1 | acc_x_band_high | 0.1223 | 0.1783 | 0.0653 | 0.1219 | 0.3502 | <1e-250 |
| 2 | ppg_signal_quality_mean | 0.7693 | 0.1491 | 0.7186 | 0.1415 | 0.3454 | <1e-187 |
| 3 | acc_x_band_low | 0.7237 | 0.2602 | 0.8028 | 0.2006 | 0.3255 | <1e-200 |
| 4 | acc_z_band_high | 0.1080 | 0.1613 | 0.0609 | 0.1138 | 0.3176 | <1e-203 |
| 5 | acc_x_spectral_centroid | 1.7389 | 1.6265 | 1.2568 | 1.2511 | 0.3175 | <1e-191 |
| 6 | acc_x_zcr | 0.2179 | 0.1742 | 0.1675 | 0.1454 | 0.3046 | <1e-166 |
| 7 | acc_y_min | -9.1296 | 7.1031 | -11.1787 | 7.2848 | 0.2862 | <1e-127 |
| 8 | acc_y_band_high | 0.1352 | 0.1787 | 0.0878 | 0.1384 | 0.2841 | <1e-153 |
| 9 | acc_mag_band_high | 0.2652 | 0.2380 | 0.1997 | 0.2262 | 0.2796 | <1e-128 |
| 10 | acc_z_band_low | 0.7485 | 0.2390 | 0.8095 | 0.1890 | 0.2719 | <1e-139 |
| 11 | acc_z_zcr | 0.1969 | 0.1600 | 0.1558 | 0.1306 | 0.2716 | <1e-135 |
| 12 | acc_z_spectral_centroid | 1.5938 | 1.4983 | 1.2137 | 1.1920 | 0.2698 | <1e-136 |
| 13 | ppg_hrv_pnn50 | 0.7924 | 0.2071 | 0.8456 | 0.1790 | 0.2678 | <1e-124 |
| 14 | acc_z_min | -1.6488 | 6.8900 | -3.4667 | 7.1136 | 0.2611 | <1e-106 |
| 15 | acc_z_range | 9.3617 | 9.8733 | 12.0064 | 10.7986 | 0.2600 | <1e-100 |
| 16 | acc_x_spectral_entropy | 4.3933 | 1.4879 | 4.0204 | 1.3386 | 0.2585 | <1e-114 |
| 17 | ppg_signal_quality_std | 0.1591 | 0.0957 | 0.1829 | 0.0904 | 0.2531 | <1e-103 |
| 18 | acc_z_std | 1.6498 | 1.6697 | 2.0700 | 1.6647 | 0.2519 | <1e-101 |
| 19 | light_log_mean | 3.1639 | 2.5049 | 3.7731 | 2.4044 | 0.2463 | <1e-99 |
| 20 | acc_x_min | -3.3774 | 8.5222 | -5.4443 | 8.5213 | 0.2425 | <1e-94 |
| 21 | acc_y_spectral_centroid | 1.8825 | 1.5714 | 1.5223 | 1.3239 | 0.2405 | <1e-104 |
| 22 | acc_mag_min | 7.5629 | 2.2553 | 7.0317 | 2.1703 | 0.2383 | <1e-93 |
| 23 | acc_y_zcr | 0.2262 | 0.1641 | 0.1889 | 0.1400 | 0.2379 | <1e-101 |
| 24 | acc_y_p25 | -5.0312 | 4.2908 | -5.9824 | 3.5362 | 0.2338 | <1e-100 |
| 25 | acc_x_dom_freq | 0.7089 | 1.6984 | 0.3615 | 0.9205 | 0.2319 | <1e-129 |
| 26 | acc_y_peak_amp | 10.1933 | 6.2404 | 11.6919 | 7.0249 | 0.2307 | <1e-78 |
| 27 | acc_y_band_low | 0.7005 | 0.2465 | 0.7546 | 0.2060 | 0.2305 | <1e-97 |
| 28 | acc_x_p25 | 0.1719 | 5.5246 | -1.0729 | 5.3567 | 0.2275 | <1e-85 |
| 29 | acc_z_spectral_entropy | 4.3589 | 1.3866 | 4.0665 | 1.2519 | 0.2173 | <1e-82 |
| 30 | acc_mag_range | 7.0333 | 9.0409 | 9.1064 | 10.6485 | 0.2165 | <1e-67 |

**Breakdown**: 20 Accelerometer, 6 PPG, 4 Light features in top 30.

## Results — LOSO-CV (38 subjects, top-30 features)

| Model | Accuracy | Balanced Acc | F1 | Precision | Recall | AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.5667 | 0.5761 | 0.4714 | 0.3945 | 0.6507 | 0.6108 |
| Random Forest | 0.6636 | 0.5552 | 0.3111 | 0.4636 | 0.2584 | 0.6353 |
| XGBoost | 0.6663 | 0.5403 | 0.2450 | 0.4713 | 0.1829 | 0.6414 |

**Best**: XGBoost AUC **0.6414**, Random Forest AUC **0.6353**, LR Balanced Accuracy **0.5761**

*All metrics are macro-averaged across 38 LOSO-CV folds.*
*Features extracted from first 16s of Accelerometer, PPG (NeuroKit2), and Light sensors.*
*Random guessing baseline: Balanced Accuracy = 0.50, AUC = 0.50.*

## LLM-Based Experiments (ConSensus-Style Multi-Agent)

Replication of the [ConSensus](https://arxiv.org/abs/2601.06453) multi-agent framework for multimodal sensing:
- **Modality Agents**: One LLM per sensor (ACC, PPG, Light), each produces prediction + reasoning
- **Statistical Fusion**: Majority vote of 3 modality agents
- **Semantic Fusion**: LLM aggregates cross-modal reasoning
- **Hybrid Fusion**: LLM arbitrates between semantic and statistical outputs (ConSensus approach)
- **Evaluation**: LOSO-CV (38 subjects), 1-shot in-context learning, subsampled ~100 per subject

### API Results (Claude Haiku)

| Method | Balanced Acc | Std | Notes |
|---|---|---|---|
| Single-Agent 0-shot | 0.4992 | 0.0370 | All modalities in one prompt |
| Single-Agent 1-shot | 0.5292 | 0.0694 | +1 example per class |
| ACC Agent only | 0.4876 | 0.0758 | 20 accelerometer features |
| PPG Agent only | 0.4924 | 0.0563 | 6 PPG features |
| Light Agent only | 0.5387 | 0.0879 | 4 light features, best individual agent |
| Statistical Fusion (majority vote) | 0.5089 | 0.0759 | Vote of 3 agents |
| Semantic Fusion | 0.5113 | 0.0808 | LLM cross-modal reasoning |
| **Hybrid Fusion (ConSensus)** | **0.5121** | **0.0791** | Semantic + statistical combined |

### Local Results (Qwen2.5-7B-Instruct)

| Method | Balanced Acc | Std | Notes |
|---|---|---|---|
| Single-Agent 1-shot | 0.5004 | 0.0024 | Log-prob scoring, full dataset |
| ACC Agent (subsampled) | 0.5126 | — | Generation-based |
| PPG Agent (subsampled) | 0.5111 | — | Generation-based |
| Light Agent (subsampled) | 0.5347 | — | Best individual agent |
| Statistical Fusion (subsampled) | 0.5253 | — | Majority vote |
| Hybrid Fusion (subsampled) | 0.5142 | — | ConSensus approach |

### Key Findings

1. **ML baselines outperform LLM approaches** on this dataset. Logistic Regression (0.5761) beats the best LLM method (Light Agent 0.5387, Single-Agent 1-shot 0.5292).
2. **Multi-agent fusion does not improve over single-agent** — opposite of ConSensus paper findings. Hybrid Fusion (0.5121) < Single-Agent 1-shot (0.5292).
3. **Light sensor is the most informative modality** for LLMs (0.5387), despite having only 4 features.
4. **Small local models (7B) fail** — Qwen-7B balanced accuracy ~0.50 (random), consistent with ConSensus paper's Llama-3.1-8B result (~26% accuracy).
5. **Possible reasons for lower LLM performance**: (a) binary task with weak signal (max Cohen's d = 0.35), (b) small model (Haiku) vs GPT-4o used in paper, (c) our LOSO-CV is stricter than paper's single split.

*ConSensus paper reference: Yoon et al., "ConSensus: Multi-Agent Collaboration for Multimodal Sensing," arXiv:2601.06453, 2026.*
