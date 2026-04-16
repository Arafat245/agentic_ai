# Social Interaction Detection from Smartwatch Sensor Data

🎥 **Presentation Video**: [Watch on SharePoint](https://myuva-my.sharepoint.com/:v:/r/personal/jgh6ds_virginia_edu/Documents/Documents/Zoom/2026-04-16%2011.38.48%20Arafat%20Rahman%27s%20Personal%20Meeting%20Room/Agentic_video.mp4?csf=1&web=1&e=SxwTHO&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D)

## Task
Binary classification: predict whether a social interaction occurred (1) or not (0) from the **first 16 seconds** of a 90-second smartwatch sensor window.

## Data
- **Subjects**: 38 (PA01–PA24, PB01–PB18)
- **Samples**: 33,727 total
- **Sensors**: Accelerometer (X/Y/Z), PPG (Green, processed with NeuroKit2), Light
- **Class distribution**: 68.5% no interaction (0), 31.5% interaction (1)
- **Evaluation**: Leave-One-Subject-Out Cross-Validation (LOSO-CV), 38 folds

## Results Summary — LOSO-CV (38 subjects)

| Model Type | Methods | BA | Precision | Recall | F1 score |
|------------|---------|-----|-----------|--------|----------|
| Classical ML | LR (Logistic Regression) | 0.5561 | 0.3945 | 0.6507 | 0.4714 |
| Classical ML | RF (Random Forest) | 0.5552 | 0.4636 | 0.2584 | 0.3111 |
| Classical ML | XGBoost | 0.5403 | 0.4713 | 0.1829 | 0.2450 |
| Deep Learning | TCN | 0.5163 | 0.3424 | 0.9500 | 0.4803 |
| Deep Learning | LSTM | 0.5252 | 0.3466 | 0.9128 | 0.4785 |
| Deep Learning | Transformer | 0.5506 | 0.3621 | 0.7988 | 0.4736 |
| LLM (3B) | Llama-3.2-3B | 0.5479 | 0.5463 | 0.7726 | 0.6244 |
| LLM (7B) | Qwen2.5-7B | 0.5347 | 0.5149 | 0.6726 | 0.5662 |
| LLM (1B) | OLMo-1B | 0.5353 | 0.5328 | 0.7579 | 0.5950 |
| Agentic | **ReAct** | **0.5695** | **0.6005** | **0.4432** | **0.4918** |

**BA** = Balanced Accuracy. All metrics macro-averaged across 38 LOSO-CV folds. Random guess baseline: BA = 0.50.

## How to Reproduce the Best ReAct Result (BA = 0.5695)

This section documents the full pipeline to reproduce the best ReAct variant (**LR + RF + Transformer + Light**), step by step. All paths below are relative to the project root (`/mnt/sdb/arafat/agentic_ai/project/`).

### Overview

The ReAct pipeline has five sequential stages:

1. **Extract 107 hand-crafted features** from raw sensor data (ACC, PPG, Light)
2. **Generate per-modality text descriptions** (used as the Light tool input)
3. **Train and save Transformer predictions** (used as one of the ReAct tools)
4. **Train ReAct agent with LR + RF + Transformer + Light tools** and evaluate with LOSO-CV
5. **Compute final metrics** from per-sample predictions

Each stage reads outputs from the previous one.

---

### Stage 1 — Feature Extraction

**Script**: `temp_exps/phase1_data_pipeline.py`

**Input files**:
- `labels_for_non_acoustic_model.pkl` — 33,727 labels with columns `P_ID`, `category`, `Sensor_start_time`
- `Processed data/{P_ID}/Smartwatch_AccelerometerDatum.pkl` — ACC raw (X, Y, Z, T)
- `Processed data/{P_ID}/Smartwatch_PPG_Health_SDK.pkl` — PPG raw (PPG Green, T)
- `Processed data/{P_ID}/Smartwatch_LightDatum.pkl` — Light raw (Light, T)

**What it does**:
- For each label, extracts the 16-second window starting at `Sensor_start_time` from each sensor file
- Computes 107 features: time-domain, frequency-domain, band-power, spectral, HRV (PPG via NeuroKit2), cross-axis correlations

**Output**:
- `temp_exps/all_subjects_features.csv` — 33,727 rows × (107 features + `P_ID` + `category` + `Sensor_start_time`)

**Sample output row**:
```
P_ID=PA01, category=0, acc_x_band_high=0.123, acc_z_zcr=0.175,
ppg_hrv_pnn50=0.792, light_log_mean=3.16, ...
```

**Run**:
```bash
python3 temp_exps/phase1_data_pipeline.py
```

---

### Stage 2 — Generate Per-Modality Text Descriptions

**Script**: `consensus_experiments/code/phase6_modality_texts.py`

**Input files**:
- `temp_exps/all_subjects_features.csv` (from Stage 1)

**What it does**:
- Converts the numeric features of each modality into natural-language paragraphs
- Produces three text columns: `text_acc`, `text_ppg`, `text_light` (the ReAct agent uses `text_light` when it calls `get_light_text`)

**Output**:
- `temp_exps/modality_text_features.csv` — 33,727 rows × (`P_ID`, `category`, `text_acc`, `text_ppg`, `text_light`)

**Sample `text_light` output**:
```
"The ambient light during this 16-second window had a log-mean of 3.16
(moderate indoor light). Light standard deviation (log) was 0.54,
suggesting stable lighting. 8 notable changes observed."
```

**Run**:
```bash
python3 consensus_experiments/code/phase6_modality_texts.py
```

---

### Stage 3 — Train and Save Transformer Predictions

**Script**: `consensus_experiments/code/phase8_dl_models.py` (Transformer variant)

**Input files**:
- `temp_exps/all_subjects_features.csv` (features)
- Raw sensor `.pkl` files (for constructing time-series tensors)

**What it does**:
- Trains a 1D Transformer encoder on raw ACC/PPG/Light time series using LOSO-CV (38 folds)
- For each held-out subject, saves the Transformer's predicted class and probability for every sample

**Output**:
- `temp_exps/dl_transformer_per_sample.csv` — columns: `P_ID`, `pred`, `prob` (one row per sample)

**Sample output row**:
```
P_ID=PA01, pred=1, prob=0.612
```

**Run**:
```bash
python3 consensus_experiments/code/phase8_dl_models.py
```

---

### Stage 4 — ReAct Agent with LR + RF + Transformer + Light Tools

**Script**: `consensus_experiments/code/phase9_react_v2.py` (runs the `lr_rf_trans_light` variant)

**Input files**:
- `temp_exps/all_subjects_features.csv` — features (Stage 1)
- `temp_exps/modality_text_features.csv` — text (Stage 2)
- `temp_exps/dl_transformer_per_sample.csv` — Transformer predictions (Stage 3)
- `.env` — must contain `OPENAI_API_KEY` (GPT-4o-mini is used as the agent LLM)

**What it does (per LOSO fold, 38 folds total)**:
1. Hold out 1 subject as test, train on remaining 37 subjects
2. **Train ML tools** on 37-subject training set:
   - `lr_predict`: Logistic Regression (`class_weight='balanced'`, `C=1.0`) on 30 top features
   - `rf_predict`: Random Forest (`n_estimators=200`, `max_depth=20`, `class_weight='balanced'`)
3. **Load Transformer predictions** from the saved lookup (Stage 3)
4. Subsample 25 positive + 25 negative = **50 test samples per subject** for efficiency
5. For each test sample, run the ReAct loop (max 3 steps):
   - **Step 0**: Agent calls `lr_predict` → receives LR probability
   - **Step 1**: If LR probability > 0.60 → agent predicts directly. Otherwise agent calls `rf_predict`, `transformer_predict`, or `get_light_text` to corroborate
   - **Step 2**: Agent commits to a final prediction (`interaction` or `no_interaction`)
6. Save trace and prediction for each sample

**ReAct System Prompt (simplified)**:
```
You are an AI agent detecting social interactions from smartwatch sensor data.
Tools:
  1. lr_predict          — Logistic Regression with probability (best ML baseline)
  2. rf_predict          — Random Forest with probability (different error pattern than LR)
  3. transformer_predict — Transformer deep-learning prediction with probability
  4. get_light_text      — Ambient light sensor description in natural language

Strategy:
  - ALWAYS call lr_predict first
  - If LR probability > 0.60, TRUST it and predict
  - If LR is uncertain (0.50–0.60), call ONE more tool (rf_predict, transformer_predict,
    or get_light_text) to verify, then decide
  - Only override LR if another tool STRONGLY disagrees
  - Max 3 tool calls

Respond in JSON: {"tool_call", "reasoning", "confidence", "prediction"}
```

**Sample ReAct trace (for subject PA01, one sample)**:

*Step 0 — Agent calls `lr_predict`*:
```json
{
  "tool_call": "lr_predict",
  "reasoning": "Starting with the most accurate tool, lr_predict, to assess the smartwatch sample.",
  "confidence": 0.0,
  "prediction": null
}
```
*Tool result*: `"LR predicts: interaction (probability: 0.563). no_interaction=0.437, interaction=0.563"`

*Step 1 — LR uncertain (0.50–0.60), agent calls `transformer_predict`*:
```json
{
  "tool_call": "transformer_predict",
  "reasoning": "The LR probability is between 0.50 and 0.60, indicating uncertainty. I will call transformer_predict to corroborate using the DL model.",
  "confidence": 0.563,
  "prediction": null
}
```
*Tool result*: `"Transformer (DL) predicts: interaction (probability: 0.612)"`

*Step 2 — Final decision*:
```json
{
  "tool_call": null,
  "reasoning": "Both LR (0.563) and Transformer (0.612) agree on interaction. I am confident in the prediction.",
  "confidence": 0.612,
  "prediction": "interaction"
}
```

**Output files**:
- `temp_exps/results_react_lr_rf_trans_light.csv` — per-subject LOSO metrics (38 rows, one per subject)
- `temp_exps/react_lr_rf_trans_light_per_sample.csv` — per-sample predictions and full JSON traces (~1,900 rows)

**Sample row of `results_react_lr_rf_trans_light.csv`**:
```
subject=PA01, accuracy=0.46, balanced_accuracy=0.46, f1=0.4490, precision=0.4583,
recall=0.44, auc=, n_test=50, n_pos=25, avg_steps=2.59
```

**Sample row of `react_lr_rf_trans_light_per_sample.csv`**:
```
P_ID=PA01, true_label=1, prediction=1, confidence=0.612, n_steps=3,
trace=[{"step":0, "tool_call":"lr_predict", ...}, {...}, {...}]
```

**Run**:
```bash
cd consensus_experiments/code
python3 phase9_react_v2.py
```
This script runs three ReAct variants sequentially; the `lr_rf_trans_light` variant is the best.

---

### Stage 5 — Final Metrics (Aggregated Across 38 Subjects)

**Input**: `temp_exps/results_react_lr_rf_trans_light.csv`

**Aggregation**: Macro-average metrics across 38 LOSO folds (each row is one subject).

**Final output — the best ReAct result**:
| Metric | Value |
|--------|-------|
| Balanced Accuracy | **0.5695** |
| Accuracy | 0.5695 |
| F1 | 0.4918 |
| Precision | 0.6005 |
| Recall | 0.4432 |
| Avg. tool calls per sample | 2.59 |

**Quick aggregation snippet**:
```python
import pandas as pd
df = pd.read_csv('temp_exps/results_react_lr_rf_trans_light.csv')
print(df[['balanced_accuracy','f1','precision','recall']].mean())
```

---

### Summary of File Flow

```
Raw sensor .pkl files
         │
         ▼  [Stage 1] phase1_data_pipeline.py
all_subjects_features.csv  (107 features × 33,727 rows)
         │
         ├─▶  [Stage 2] phase6_modality_texts.py
         │        │
         │        ▼
         │   modality_text_features.csv  (text per modality)
         │
         ├─▶  [Stage 3] phase8_dl_models.py (Transformer)
         │        │
         │        ▼
         │   dl_transformer_per_sample.csv  (pred, prob per sample)
         │
         ▼  [Stage 4] phase9_react_v2.py (lr_rf_trans_light variant)
results_react_lr_rf_trans_light.csv        (per-subject metrics)
react_lr_rf_trans_light_per_sample.csv     (per-sample traces)
         │
         ▼  [Stage 5] aggregation
Final: BA = 0.5695
```

---

## Analysis

### Key Findings

1. **ReAct achieves the best performance.** The ReAct agent reaches a balanced accuracy of **0.5695**, outperforming all other model families. Its ability to reason over ML tool outputs and select between classifiers provides an advantage over any single standalone method.

2. **Classical ML baselines are competitive.** Logistic Regression (0.5561 BA) and Random Forest (0.5552 BA) perform nearly identically, both beating all pure deep learning and LLM-only approaches except Transformer. This suggests the hand-crafted sensor features (time-domain, frequency-domain, spectral) already capture most of the discriminative signal in the first 16 seconds of data.

3. **Deep learning struggles to generalize across unseen subjects.** Despite raw sensor input, TCN (0.5163), LSTM (0.5252), and Transformer (0.5506) all perform below or at par with simple LR. TCN and LSTM show high recall (>0.91) but very low precision (~0.34), indicating they default to the majority-positive prediction pattern. Transformer is the best DL variant but still falls short of ReAct and classical ML.

4. **LLMs show different trade-offs than ML.** The LLM-based Light Agents (Llama-3.2-3B, Qwen2.5-7B, OLMo-1B) produce the highest F1 scores (0.57–0.62) due to their strong recall (0.67–0.77), but their balanced accuracy (0.53–0.55) lags behind classical ML. LLMs appear to reason broadly about environmental context from light readings but cannot exploit the finer-grained accelerometer/PPG structure as effectively as feature-engineered classifiers.

5. **Model size does not guarantee better performance in LLMs.** OLMo-1B (0.5353) matches Qwen-7B (0.5347), and Llama-3.2-3B (0.5479) slightly exceeds both despite being smaller than Qwen. This indicates that scale alone is not a reliable predictor of downstream sensing accuracy — training data and model design matter more than parameter count for this task.

6. **All methods remain far from strong performance.** Even the best model reaches only BA ≈ 0.57, far below practical deployment thresholds. The task is inherently difficult: 16 seconds of smartwatch accelerometer, PPG, and light signals carry only a weak signature of social interaction, and LOSO-CV enforces generalization to entirely unseen subjects with their own behavioral baselines.

### Takeaway

ReAct yields the strongest results, but the gap over plain Logistic Regression is small (0.5695 vs 0.5561). For production use, Logistic Regression offers the best accuracy-per-compute trade-off. Further improvements will likely require longer sensor windows, subject-adaptive calibration, or sensor-language pretraining (e.g., SensorLM-style alignment) rather than scaling up model capacity alone.
