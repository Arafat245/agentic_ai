"""
Stacking classifier: LR + RF only (no XGBoost). Fast.
LOSO-CV with subsampled test (50/subject).
"""

import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from pathlib import Path

R = Path('temp_exps')
SEED = 42
N_PER_CLASS = 25

TOP_FEATURES = [
    'acc_x_band_high','acc_x_band_low','acc_z_band_high','acc_x_spectral_centroid','acc_x_zcr','acc_y_min',
    'acc_y_band_high','acc_mag_band_high','acc_z_band_low','acc_z_zcr','acc_z_spectral_centroid','acc_z_min',
    'acc_z_range','acc_x_spectral_entropy','acc_z_std','acc_x_min','acc_y_spectral_centroid','acc_mag_min',
    'acc_y_zcr','acc_mag_range','ppg_signal_quality_mean','ppg_hrv_pnn50','ppg_signal_quality_std',
    'ppg_hrv_pnn20','ppg_zcr','ppg_peak_rate','light_log_mean','light_kurtosis','light_log_std','light_n_changes',
]

def subsample(test_df, seed):
    rng = np.random.RandomState(seed)
    parts = []
    for cat in [0, 1]:
        pool = test_df[test_df['category'] == cat]
        n = min(N_PER_CLASS, len(pool))
        if n > 0:
            parts.append(pool.sample(n=n, random_state=rng))
    return pd.concat(parts).sort_index()

def main():
    df = pd.read_csv(R / 'all_subjects_features.csv')
    df['category'] = df['category'].astype(int)
    df[TOP_FEATURES] = df[TOP_FEATURES].fillna(0)
    subjects = sorted(df['P_ID'].unique())

    base_models = {
        'LR': lambda: LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED, C=1.0),
        'RF': lambda: RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=SEED, n_jobs=4),
    }

    t0 = time.time()
    all_results = []
    all_preds = []

    for fold_i, test_subj in enumerate(subjects):
        train = df[df['P_ID'] != test_subj]
        test = df[df['P_ID'] == test_subj]
        test_sub = subsample(test, SEED + fold_i)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[TOP_FEATURES].values)
        X_test = scaler.transform(test_sub[TOP_FEATURES].values)
        y_train = train['category'].values
        y_test = test_sub['category'].values

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        train_meta = np.zeros((len(X_train), len(base_models)))
        test_meta = np.zeros((len(X_test), len(base_models)))

        for i, (name, model_fn) in enumerate(base_models.items()):
            model = model_fn()
            train_probs = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
            train_meta[:, i] = train_probs
            model = model_fn()
            model.fit(X_train, y_train)
            test_meta[:, i] = model.predict_proba(X_test)[:, 1]

        meta = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
        meta.fit(train_meta, y_train)

        y_pred = meta.predict(test_meta)
        y_prob = meta.predict_proba(test_meta)[:, 1]

        ba = balanced_accuracy_score(y_test, y_pred)
        try: auc = roc_auc_score(y_test, y_prob)
        except: auc = np.nan

        all_results.append({
            'subject': test_subj, 'balanced_accuracy': ba,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'auc': auc, 'n_test': len(y_test), 'n_pos': int(y_test.sum()),
        })
        for i in range(len(y_test)):
            all_preds.append({'P_ID': test_subj, 'true_label': int(y_test[i]), 'pred': int(y_pred[i]), 'prob': float(y_prob[i])})

        if (fold_i + 1) % 10 == 0:
            print(f'  {fold_i+1}/38 done', flush=True)

    rdf = pd.DataFrame(all_results)
    rdf.to_csv(R / 'results_stacked_lr_rf.csv', index=False)
    pd.DataFrame(all_preds).to_csv(R / 'stacked_lr_rf_per_sample.csv', index=False)

    elapsed = time.time() - t0
    ba = rdf['balanced_accuracy'].mean()
    auc_val = rdf['auc'].mean()
    print(f'Stacked (LR+RF):     BalAcc={ba:.4f}, AUC={auc_val:.4f} ({elapsed:.1f}s)')
    print(f'LR alone:            BalAcc=0.5653, AUC=0.6061')

if __name__ == '__main__':
    main()
