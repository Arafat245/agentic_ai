# Social Interaction Detection from Smartwatch Sensor Data

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
| LLM (3B) | Llama-3.2-3B (Light Agent) | 0.5479 | 0.5463 | 0.7726 | 0.6244 |
| LLM (7B) | Qwen2.5-7B (Light Agent) | 0.5347 | 0.5149 | 0.6726 | 0.5662 |
| LLM (1B) | OLMo-1B (Light Agent) | 0.5353 | 0.5328 | 0.7579 | 0.5950 |
| Agentic | **ReAct** | **0.5695** | **0.6019** | **0.4389** | **0.4898** |

**BA** = Balanced Accuracy. All metrics macro-averaged across 38 LOSO-CV folds. Random guess baseline: BA = 0.50.

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
