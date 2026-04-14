# CLAUDE.md — Project Guidelines

## Project Overview

**Task**: Social interaction detection from smartwatch sensor data.
- **Category 0** = No social interaction
- **Category 1** = Social interaction occurred
- Predict whether a social interaction is happening using the **first 16 seconds** of each 90-second sensor window (not the full 90s).
- Data collected via EMA (Ecological Momentary Assessment) study with 38 subjects (PA01–PA24, PB01–PB18).

## Data

- **Processed data/**: Per-subject folders, each containing `.pkl` files for various smartwatch sensors.
- **labels_for_non_acoustic_model.pkl**: Labels dataframe (33,727 rows) with columns `P_ID`, `category` (binary: 0/1, stored as strings), `Sensor_start_time` (timezone-aware, stored as strings).
- Each sample corresponds to a **90-second sensor window** (consecutive `Sensor_start_time` values are ~90s apart). However, we only use the **first 16 seconds** of each window for prediction.
- Class distribution is imbalanced: ~68.5% class 0 (no interaction), ~31.5% class 1 (interaction).
- Samples per subject range from 290 (PA11) to 1,585 (PB16).
- **papers/**: Reference papers (PDFs).

## Sensor Data Rules

- **Use only these 3 sensors**: Accelerometer (`Smartwatch_AccelerometerDatum.pkl`), PPG (`Smartwatch_PPG_Health_SDK.pkl`), and Light (`Smartwatch_LightDatum.pkl`).
- Do NOT use audio, heart rate, step count, gravity, or any other sensor data unless explicitly asked.

## Evaluation Rules

- **Always use Leave-One-Subject-Out Cross-Validation (LOSO-CV)**. Every experiment must evaluate by training on N-1 subjects and testing on the held-out subject, iterating over all subjects.
- Never use random train/test splits — data from the same subject must never appear in both train and test sets.
- Report per-subject results and overall aggregated metrics.

## Documentation

- **Always create or update `README.md`** with the latest experimental results after each experiment.
- Include: model name, features used, LOSO-CV metrics (accuracy, F1, precision, recall, AUC), and any relevant notes.
- Keep a results table in README.md so progress is trackable over time.

## Experiment Workflow

- **Run all new experiments in `temp_exps/` first.** Any new Python scripts, notebooks, or trial code goes here initially.
- Only promote a script/approach to the main project folder once it has proven to produce better results than the current best.
- This keeps the main folder clean — only validated, best-performing code lives there.

## Code Conventions

- Use Python with standard ML libraries (scikit-learn, pandas, numpy, pytorch/tensorflow as needed).
- Use reproducible random seeds where applicable.
- Keep code clean and modular — separate data loading, feature extraction, model training, and evaluation.
