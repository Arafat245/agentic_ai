"""
Phase 8 — TCN, LSTM, and Transformer models for raw sensor time series.
LOSO-CV on 38 subjects. Input: 16-second windows (320 timesteps × 5 channels).
Saves all per-subject metrics, per-sample predictions, and model outputs.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
from pathlib import Path
import time
import pickle
import sys
import warnings
import math

warnings.filterwarnings('ignore')

PROJECT_DIR = Path('/mnt/sdb/arafat/agentic_ai/project')
RESULTS_DIR = PROJECT_DIR / 'temp_exps'
GPU_ID = 0  # CUDA device 0 = A6000 (free, Llama uses CUDA 1)
SEED = 42
SEQ_LEN = 320
N_CHANNELS = 5  # 3 acc + 1 ppg + 1 light

# Training hyperparameters
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
PATIENCE = 7  # Early stopping


# ===================== MODELS =====================

class TCN(nn.Module):
    """Temporal Convolutional Network with dilated causal convolutions."""
    def __init__(self, n_channels=N_CHANNELS, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3, dilation=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=4, dilation=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class LSTMModel(nn.Module):
    """Bidirectional LSTM for time series classification."""
    def __init__(self, n_channels=N_CHANNELS, hidden_size=64, n_layers=2, n_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels, hidden_size=hidden_size,
            num_layers=n_layers, batch_first=True,
            bidirectional=True, dropout=0.3,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        # x: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.permute(0, 2, 1)
        out, (h_n, _) = self.lstm(x)
        # Use last hidden states from both directions
        h_cat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        h_cat = self.dropout(h_cat)
        return self.fc(h_cat)


class TransformerModel(nn.Module):
    """Transformer encoder for time series classification."""
    def __init__(self, n_channels=N_CHANNELS, d_model=64, nhead=4, n_layers=2, n_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=SEQ_LEN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.3, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        # Pool over time
        x = x.permute(0, 2, 1)  # (batch, d_model, seq_len)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ===================== TRAINING =====================

def get_class_weights(y):
    """Compute class weights for imbalanced data."""
    counts = np.bincount(y)
    weights = 1.0 / counts
    return torch.FloatTensor(weights)


def get_sampler(y):
    """Weighted random sampler for balanced batches."""
    counts = np.bincount(y)
    weight_per_class = 1.0 / counts
    sample_weights = weight_per_class[y]
    return WeightedRandomSampler(sample_weights, len(y), replacement=True)


def train_one_fold(model, train_X, train_y, device, epochs=EPOCHS, patience=PATIENCE):
    """Train model with early stopping."""
    from phase8_dl_data_loader import SensorDataset

    dataset = SensorDataset(train_X, train_y)
    sampler = get_sampler(train_y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)

    class_weights = get_class_weights(train_y).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model


def evaluate_fold(model, test_X, test_y, device):
    """Evaluate model and return predictions + probabilities."""
    from phase8_dl_data_loader import SensorDataset

    dataset = SensorDataset(test_X, test_y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs[:, 1].cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    return y_pred, y_prob


def normalize(train_X, test_X):
    """Z-score normalize per channel."""
    mean = train_X.mean(axis=(0, 1), keepdims=True)
    std = train_X.std(axis=(0, 1), keepdims=True)
    std[std < 1e-8] = 1.0
    return (train_X - mean) / std, (test_X - mean) / std


# ===================== MAIN =====================

def run_loso_cv(model_class, model_name, all_data, device):
    """Run full LOSO-CV for a given model class."""
    subjects = sorted(all_data.keys())
    all_results = []
    all_sample_preds = []

    print(f"\n{'='*60}")
    print(f"  {model_name} — LOSO-CV ({len(subjects)} subjects)")
    print(f"{'='*60}")

    t0 = time.time()

    for fold_i, test_subj in enumerate(subjects):
        t_fold = time.time()

        # Build train/test arrays
        train_Xs, train_ys = [], []
        for subj in subjects:
            if subj != test_subj:
                X, y = all_data[subj]
                train_Xs.append(X)
                train_ys.append(y)
        train_X = np.concatenate(train_Xs)
        train_y = np.concatenate(train_ys)
        test_X, test_y = all_data[test_subj]

        # Normalize
        train_X, test_X = normalize(train_X, test_X)

        # Create model
        torch.manual_seed(SEED)
        model = model_class().to(device)

        # Train
        model = train_one_fold(model, train_X, train_y, device)

        # Evaluate
        y_pred, y_prob = evaluate_fold(model, test_X, test_y, device)

        # Metrics
        metrics = {
            'subject': test_subj,
            'accuracy': accuracy_score(test_y, y_pred),
            'balanced_accuracy': balanced_accuracy_score(test_y, y_pred),
            'f1': f1_score(test_y, y_pred, zero_division=0),
            'precision': precision_score(test_y, y_pred, zero_division=0),
            'recall': recall_score(test_y, y_pred, zero_division=0),
            'n_test': len(test_y),
            'n_pos': int(test_y.sum()),
        }
        try:
            metrics['auc'] = roc_auc_score(test_y, y_prob)
        except ValueError:
            metrics['auc'] = np.nan

        all_results.append(metrics)

        # Save per-sample predictions
        for i in range(len(test_y)):
            all_sample_preds.append({
                'P_ID': test_subj,
                'true_label': int(test_y[i]),
                'pred': int(y_pred[i]),
                'prob': float(y_prob[i]),
            })

        elapsed_fold = time.time() - t_fold
        print(f"  [{fold_i+1:2d}/38] {test_subj}: "
              f"BalAcc={metrics['balanced_accuracy']:.3f}, "
              f"F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f} "
              f"({metrics['n_test']} samples, {elapsed_fold:.1f}s)", flush=True)

        # === INCREMENTAL SAVE after each subject ===
        safe_name = model_name.lower().replace(' ', '_')
        pd.DataFrame(all_results).to_csv(RESULTS_DIR / f'results_dl_{safe_name}.csv', index=False)
        pd.DataFrame(all_sample_preds).to_csv(RESULTS_DIR / f'dl_{safe_name}_per_sample.csv', index=False)

        # Clean up
        del model
        torch.cuda.empty_cache()

    elapsed = time.time() - t0

    # Final save
    results_df = pd.DataFrame(all_results)
    safe_name = model_name.lower().replace(' ', '_')
    results_df.to_csv(RESULTS_DIR / f'results_dl_{safe_name}.csv', index=False)

    preds_df = pd.DataFrame(all_sample_preds)
    preds_df.to_csv(RESULTS_DIR / f'dl_{safe_name}_per_sample.csv', index=False)

    # Macro averages
    metric_cols = ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 'auc']
    macro = {m: results_df[m].mean() for m in metric_cols}

    print(f"\n  MACRO AVERAGES:")
    for m, v in macro.items():
        print(f"    {m:20s}: {v:.4f}")
    print(f"    {'Time':20s}: {elapsed:.1f}s ({elapsed/3600:.1f}h)")
    print(f"  Saved to results_dl_{safe_name}.csv + dl_{safe_name}_per_sample.csv")

    return results_df, macro


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    from phase8_dl_data_loader import load_all_subjects

    # Load data
    all_data = load_all_subjects()
    total = sum(len(y) for _, (_, y) in all_data.items())
    print(f"Total: {total} samples across {len(all_data)} subjects")

    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    models = [
        (TCN, "TCN"),
        (LSTMModel, "LSTM"),
        (TransformerModel, "Transformer"),
    ]

    all_macros = {}
    for model_class, model_name in models:
        results_df, macro = run_loso_cv(model_class, model_name, all_data, device)
        all_macros[model_name] = macro

    # Final comparison
    print(f"\n{'='*60}")
    print("  DEEP LEARNING SUMMARY (Balanced Accuracy)")
    print(f"{'='*60}")
    for name, macro in all_macros.items():
        print(f"  {name:15s}: {macro['balanced_accuracy']:.4f} (AUC: {macro['auc']:.4f})")
    print(f"  {'LR Baseline':15s}: 0.5761 (AUC: 0.6108)")

    # Save summary
    summary_rows = [{'Model': k, **{m: f"{v:.4f}" for m, v in macro.items()}} for k, macro in all_macros.items()]
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / 'results_dl_summary.csv', index=False)
    print(f"\nSaved summary to results_dl_summary.csv")


if __name__ == '__main__':
    main()
