"""
Phase 2: Traditional ML Baselines with LOSO-CV
- Loads extracted features from Phase 1
- Trains Random Forest, XGBoost, Logistic Regression, SVM
- Evaluates with Leave-One-Subject-Out Cross-Validation
- Saves results to README.md
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)
from pathlib import Path
import warnings
import time

warnings.filterwarnings('ignore')

PROJECT_DIR = Path('/mnt/sdb/arafat/agentic_ai/project')
FEATURES_FILE = PROJECT_DIR / 'temp_exps' / 'features_16s.pkl'
RESULTS_DIR = PROJECT_DIR / 'temp_exps'

SEED = 42


def load_features():
    df = pd.read_pickle(FEATURES_FILE)
    feat_cols = [c for c in df.columns if c not in ['P_ID', 'category', 'Sensor_start_time']]
    # Fill NaN with 0 (missing sensor windows)
    df[feat_cols] = df[feat_cols].fillna(0)
    return df, feat_cols


def loso_cv(df, feat_cols, model_fn, model_name):
    """Run Leave-One-Subject-Out CV and return results."""
    subjects = sorted(df['P_ID'].unique())
    results = []

    print(f"\n{'='*60}")
    print(f"  {model_name} — LOSO-CV ({len(subjects)} subjects)")
    print(f"{'='*60}")

    t0 = time.time()

    for i, test_subject in enumerate(subjects):
        train_mask = df['P_ID'] != test_subject
        test_mask = df['P_ID'] == test_subject

        X_train = df.loc[train_mask, feat_cols].values
        y_train = df.loc[train_mask, 'category'].values
        X_test = df.loc[test_mask, feat_cols].values
        y_test = df.loc[test_mask, 'category'].values

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train
        model = model_fn()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred.astype(float)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = np.nan

        results.append({
            'subject': test_subject,
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'f1': f1,
            'precision': prec,
            'recall': rec,
            'auc': auc,
            'n_test': len(y_test),
            'n_pos': int(y_test.sum()),
            'n_neg': int(len(y_test) - y_test.sum()),
        })

        if (i + 1) % 10 == 0 or i == len(subjects) - 1:
            print(f"  [{i+1}/{len(subjects)}] Last: {test_subject} — Acc: {acc:.3f}, BalAcc: {bal_acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

    elapsed = time.time() - t0
    results_df = pd.DataFrame(results)

    # Compute macro averages
    macro = {
        'accuracy': results_df['accuracy'].mean(),
        'balanced_accuracy': results_df['balanced_accuracy'].mean(),
        'f1': results_df['f1'].mean(),
        'precision': results_df['precision'].mean(),
        'recall': results_df['recall'].mean(),
        'auc': results_df['auc'].mean(),
    }

    print(f"\n  MACRO AVERAGES:")
    print(f"    Accuracy:          {macro['accuracy']:.4f}")
    print(f"    Balanced Accuracy: {macro['balanced_accuracy']:.4f}")
    print(f"    F1:                {macro['f1']:.4f}")
    print(f"    Precision:         {macro['precision']:.4f}")
    print(f"    Recall:            {macro['recall']:.4f}")
    print(f"    AUC:               {macro['auc']:.4f}")
    print(f"    Time:              {elapsed:.1f}s")

    return results_df, macro


def get_models():
    """Return dict of model name -> model factory function."""
    models = {
        'Random Forest': lambda: RandomForestClassifier(
            n_estimators=200, max_depth=20, class_weight='balanced',
            random_state=SEED, n_jobs=-1
        ),
        'Logistic Regression': lambda: LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=SEED, C=1.0
        ),
        'SVM (RBF)': lambda: SVC(
            kernel='rbf', class_weight='balanced', random_state=SEED,
            probability=True, C=1.0
        ),
    }

    # Try importing XGBoost
    try:
        from xgboost import XGBClassifier
        # Compute scale_pos_weight later based on data
        models['XGBoost'] = lambda: XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=SEED, n_jobs=-1, eval_metric='logloss',
            use_label_encoder=False
        )
    except ImportError:
        print("XGBoost not installed, skipping.")

    try:
        from lightgbm import LGBMClassifier
        models['LightGBM'] = lambda: LGBMClassifier(
            n_estimators=200, max_depth=10, learning_rate=0.1,
            class_weight='balanced', random_state=SEED, n_jobs=-1, verbose=-1
        )
    except ImportError:
        print("LightGBM not installed, skipping.")

    return models


def save_results(all_results, all_macros):
    """Save results summary and update README.md."""
    # Save detailed per-subject results
    for model_name, results_df in all_results.items():
        safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        results_df.to_csv(RESULTS_DIR / f'results_{safe_name}.csv', index=False)

    # Create summary table
    summary_rows = []
    for model_name, macro in all_macros.items():
        summary_rows.append({
            'Model': model_name,
            'Accuracy': f"{macro['accuracy']:.4f}",
            'Balanced Accuracy': f"{macro['balanced_accuracy']:.4f}",
            'F1': f"{macro['f1']:.4f}",
            'Precision': f"{macro['precision']:.4f}",
            'Recall': f"{macro['recall']:.4f}",
            'AUC': f"{macro['auc']:.4f}",
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / 'results_summary.csv', index=False)
    print(f"\nSummary saved to {RESULTS_DIR / 'results_summary.csv'}")

    # Update README.md
    readme_path = PROJECT_DIR / 'README.md'
    readme_content = f"""# Social Interaction Detection from Smartwatch Sensor Data

## Task
Binary classification: predict whether a social interaction occurred (1) or not (0)
from the first 16 seconds of a 90-second smartwatch sensor window.

## Data
- **Subjects**: 38 (PA01-PA24, PB01-PB18)
- **Samples**: 33,727 total
- **Sensors**: Accelerometer (X/Y/Z), PPG (Green), Light
- **Class distribution**: ~68.5% no interaction (0), ~31.5% interaction (1)
- **Evaluation**: Leave-One-Subject-Out Cross-Validation (LOSO-CV)

## Results

### Phase 2: Traditional ML Baselines (Hand-crafted Features, 16s Window)

| Model | Accuracy | Balanced Accuracy | F1 | Precision | Recall | AUC |
|-------|----------|-------------------|-----|-----------|--------|-----|
"""
    for _, row in summary_df.iterrows():
        readme_content += f"| {row['Model']} | {row['Accuracy']} | {row['Balanced Accuracy']} | {row['F1']} | {row['Precision']} | {row['Recall']} | {row['AUC']} |\n"

    readme_content += f"""
*All metrics are macro-averaged across {len(all_results[list(all_results.keys())[0]])} LOSO-CV folds.*
*Features: time-domain + frequency-domain from ACC (X/Y/Z/Mag), PPG, Light.*
"""

    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"README.md updated at {readme_path}")


def main():
    print("Loading features...")
    df, feat_cols = load_features()
    print(f"Loaded {len(df)} samples, {len(feat_cols)} features, {df['P_ID'].nunique()} subjects")
    print(f"Class distribution: {dict(df['category'].value_counts().sort_index())}")

    models = get_models()
    all_results = {}
    all_macros = {}

    for model_name, model_fn in models.items():
        results_df, macro = loso_cv(df, feat_cols, model_fn, model_name)
        all_results[model_name] = results_df
        all_macros[model_name] = macro

    save_results(all_results, all_macros)

    # Print final comparison
    print(f"\n{'='*60}")
    print("  FINAL COMPARISON")
    print(f"{'='*60}")
    summary = pd.DataFrame([
        {'Model': name, **{k: f"{v:.4f}" for k, v in macro.items()}}
        for name, macro in all_macros.items()
    ])
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
