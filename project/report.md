# Detecting Social Interactions from Smartwatch Sensors: A Short Report

## Motivation

Social interaction is a basic ingredient of well-being. Positive contact lifts mood and gives meaning to the day, while withdrawal and avoidance track closely with anxiety and depression. If we could automatically tell, from a wrist-worn device, when a person is engaged in social interaction, we could begin to build context-aware systems that respond to a user's social environment in real time — nudging, reflecting, or simply logging.

Prior work on this problem has leaned heavily on on-device audio. The SocialPulse system (Ahmed et al., 2026), the immediate inspiration for the dataset we use, reaches a balanced accuracy of roughly 90% by detecting the watch user's foreground speech from 15-second audio windows. Audio is, however, the most privacy-sensitive signal a watch can capture, and audio classifiers also dominate the device's compute budget. A natural question follows: how much of the social-interaction signal survives if we throw the microphone away and keep only the cheap, privacy-preserving motion and physiological sensors?

This project investigates exactly that. Using only accelerometer, PPG, and ambient light, and only the first 16 seconds of each 90-second window, we ask how far conventional machine learning, deep learning, large language models, and an agentic ReAct pipeline can each push the problem.

## Methods

We use the EMA deployment data of 38 participants (PA01–PA24, PB01–PB18), with 33,727 labelled windows in total. The class balance is roughly 68/32 in favour of "no interaction".

**Features.** For each 16-second window we extract 107 hand-crafted features per sample: time-domain statistics and zero-crossings of the accelerometer axes, cross-axis correlations, band-power and spectral features, HRV indices computed from PPG via NeuroKit2, and log-scale summaries of the light channel.

**Models.** We compare four families on the same windows:

1. *Classical ML* — Logistic Regression, Random Forest, XGBoost trained on the 107 features.
2. *Deep learning* — TCN, LSTM, and a small 1D Transformer encoder trained directly on the raw multi-channel time series.
3. *LLM zero/few-shot* — Llama-3.2-3B, Qwen2.5-7B, and OLMo-1B prompted with natural-language descriptions of the per-modality features.
4. *Agentic ReAct* — A GPT-4o-mini agent that, for each sample, can call up to three of four tools (`lr_predict`, `rf_predict`, `transformer_predict`, `get_light_text`) and must decide when to stop and commit to a prediction. The ML tools are re-trained inside every LOSO fold; the Transformer is pre-computed per fold and looked up.

**Evaluation.** We use Leave-One-Subject-Out Cross-Validation throughout, with no subject appearing in both training and test for any fold. Metrics are macro-averaged across the 38 folds.

## Evaluation

The headline numbers across all 38 LOSO-CV folds are summarised below (balanced accuracy is the most informative metric given the class imbalance).

| Family | Best model | Balanced Accuracy | F1 |
|---|---|---|---|
| Classical ML | Logistic Regression | 0.5561 | 0.4714 |
| Deep learning | Transformer | 0.5506 | 0.4736 |
| LLM | Llama-3.2-3B | 0.5479 | 0.6244 |
| Agentic | **ReAct (LR + RF + Transformer + Light)** | **0.5695** | **0.4918** |

A few observations stand out. ReAct gives the best balanced accuracy, but its margin over plain Logistic Regression is small (about 1.3 points). Deep models lean toward predicting the positive class — TCN and LSTM reach recall above 0.91 but precision near 0.34, a clear sign of collapse rather than discrimination. The LLM-based light agents earn the highest F1 scores via aggressive positive predictions, but their balanced accuracy lags the classical baselines. Model scale among LLMs is not predictive of accuracy: the 1B OLMo matches the 7B Qwen.

Across the board, every method sits in a narrow band between 0.51 and 0.57 balanced accuracy. That band is well above random, but well below the ~0.90 reported in SocialPulse with audio.

## Conclusions

The clearest lesson is a negative one: 16 seconds of accelerometer, PPG, and light carry only a weak signature of social interaction once the model has to generalise to an unseen subject. The dominant variability in these signals is between people — their resting heart rates, their workspaces, the way they hold their wrist — and 16 seconds is too short to wash that out.

Within that ceiling, the ranking is intuitive. Hand-crafted features plus Logistic Regression remain remarkably hard to beat; the discriminative signal that exists is mostly linear in well-chosen features. Larger deep models do not help, partly because the dataset is small relative to their capacity and partly because LOSO-CV punishes any subject-specific shortcut they learn. LLMs are interesting but mis-calibrated: they reason fluently about light and motion descriptions but commit too readily to the positive class.

The agentic ReAct pipeline is the best performer, and the reason matters. It does not introduce new information; it composes existing classifiers and decides, per sample, when one tool is confident enough to trust. The gain is modest but consistent — evidence that meta-reasoning over heterogeneous predictors is a worthwhile direction, even when no individual predictor is strong.

For follow-up work, the most promising levers are not bigger models but more signal: longer windows, light-touch per-subject calibration, or sensor-language pretraining that lets the model exploit the structure of accelerometer and PPG the way SocialPulse exploits audio.
