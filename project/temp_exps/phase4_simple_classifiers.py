"""
Phase 4: Simple Classifiers on Most Discriminative Features (LOSO-CV)
- Uses top features selected by Cohen's d effect size
- Tests Random Forest, Logistic Regression, SVM, XGBoost, LightGBM
- Leave-One-Subject-Out Cross-Validation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
from pathlib import Path
import time
import warnings

warnings.filterwarnings('ignore')

PROJECT_DIR = Path('/mnt/sdb/arafat/agentic_ai/project')
FEATURES_FILE = PROJECT_DIR / 'temp_exps' / 'all_subjects_features.csv'
RESULTS_DIR = PROJECT_DIR / 'temp_exps'
SEED = 42

# Top 30 most discriminative features by Cohen's d
TOP_FEATURES = [
    # ACC (20 features)
    'acc_x_band_high', 'acc_x_band_low', 'acc_z_band_high',
    'acc_x_spectral_centroid', 'acc_x_zcr', 'acc_y_min',
    'acc_y_band_high', 'acc_mag_band_high', 'acc_z_band_low',
    'acc_z_zcr', 'acc_z_spectral_centroid', 'acc_z_min',
    'acc_z_range', 'acc_x_spectral_entropy', 'acc_z_std',
    'acc_x_min', 'acc_y_spectral_centroid', 'acc_mag_min',
    'acc_y_zcr', 'acc_mag_range',
    # PPG (6 features)
    'ppg_signal_quality_mean', 'ppg_hrv_pnn50',
    'ppg_signal_quality_std', 'ppg_hrv_pnn20',
    'ppg_zcr', 'ppg_peak_rate',
    # Light (4 features)
    'light_log_mean', 'light_kurtosis', 'light_log_std', 'light_n_changes',
]


def load_features():
    df = pd.read_csv(FEATURES_FILE)
    df['category'] = df['category'].astype(int)
    # Fill NaN with 0
    df[TOP_FEATURES] = df[TOP_FEATURES].fillna(0)
    return df


def get_models():
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

    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = lambda: XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=SEED, n_jobs=-1, eval_metric='logloss',
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


def loso_cv(df, feat_cols, model_fn, model_name):
    subjects = sorted(df['P_ID'].unique())
    results = []

    print(f"\n{'='*60}")
    print(f"  {model_name} — LOSO-CV ({len(subjects)} subjects, {len(feat_cols)} features)")
    print(f"{'='*60}")

    t0 = time.time()

    for i, test_subject in enumerate(subjects):
        train_mask = df['P_ID'] != test_subject
        test_mask = df['P_ID'] == test_subject

        X_train = df.loc[train_mask, feat_cols].values
        y_train = df.loc[train_mask, 'category'].values
        X_test = df.loc[test_mask, feat_cols].values
        y_test = df.loc[test_mask, 'category'].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = model_fn()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred.astype(float)

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
            'subject': test_subject, 'accuracy': acc,
            'balanced_accuracy': bal_acc, 'f1': f1,
            'precision': prec, 'recall': rec, 'auc': auc,
            'n_test': len(y_test), 'n_pos': int(y_test.sum()),
            'n_neg': int(len(y_test) - y_test.sum()),
        })

        if (i + 1) % 10 == 0 or i == len(subjects) - 1:
            print(f"  [{i+1}/{len(subjects)}] Last: {test_subject} — "
                  f"Acc: {acc:.3f}, BalAcc: {bal_acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

    elapsed = time.time() - t0
    results_df = pd.DataFrame(results)

    macro = {m: results_df[m].mean() for m in
             ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 'auc']}

    print(f"\n  MACRO AVERAGES:")
    for m, v in macro.items():
        print(f"    {m:20s}: {v:.4f}")
    print(f"    {'Time':20s}: {elapsed:.1f}s")

    return results_df, macro


def main():
    print("Loading features...")
    df = load_features()
    print(f"Loaded {len(df)} samples, {df['P_ID'].nunique()} subjects")
    print(f"Using {len(TOP_FEATURES)} most discriminative features")
    print(f"Class distribution: {dict(df['category'].value_counts().sort_index())}")

    # Also run with ALL 164 features for comparison
    all_feat_cols = [c for c in df.columns if c not in ['P_ID', 'category', 'Sensor_start_time']]
    df[all_feat_cols] = df[all_feat_cols].fillna(0)

    models = get_models()
    all_results = {}
    all_macros = {}

    # Run with top discriminative features
    print(f"\n{'#'*60}")
    print(f"  PART 1: Top {len(TOP_FEATURES)} Discriminative Features")
    print(f"{'#'*60}")

    for model_name, model_fn in models.items():
        results_df, macro = loso_cv(df, TOP_FEATURES, model_fn, model_name)
        key = f"{model_name} (top-{len(TOP_FEATURES)})"
        all_results[key] = results_df
        all_macros[key] = macro

    # Run with all 164 features for comparison
    print(f"\n{'#'*60}")
    print(f"  PART 2: All {len(all_feat_cols)} Features")
    print(f"{'#'*60}")

    for model_name, model_fn in models.items():
        results_df, macro = loso_cv(df, all_feat_cols, model_fn, model_name)
        key = f"{model_name} (all-{len(all_feat_cols)})"
        all_results[key] = results_df
        all_macros[key] = macro

    # Save results
    for key, results_df in all_results.items():
        safe_name = key.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        results_df.to_csv(RESULTS_DIR / f'results_{safe_name}.csv', index=False)

    # Print final comparison
    print(f"\n{'='*60}")
    print("  FINAL COMPARISON")
    print(f"{'='*60}")
    summary_rows = []
    for key, macro in all_macros.items():
        summary_rows.append({'Model': key, **{k: f"{v:.4f}" for k, v in macro.items()}})
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    summary_df.to_csv(RESULTS_DIR / 'results_discriminative_summary.csv', index=False)
    print(f"\nSaved to {RESULTS_DIR / 'results_discriminative_summary.csv'}")


if __name__ == '__main__':
    main()
