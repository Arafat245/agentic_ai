"""
Phase 6 — Compile comprehensive report with all outputs for course presentation.
Run this AFTER all experiments complete.
Generates: summary tables, example predictions, per-sample exports, and plots.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_DIR = Path('/mnt/sdb/arafat/agentic_ai/project')
TEMP_EXPS = PROJECT_DIR / 'temp_exps'
REPORT_DIR = PROJECT_DIR / 'consensus_experiments'


def load_pkl(path):
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def compile_all_results():
    """Load all per-subject result CSVs and build comparison table."""
    result_files = {
        # ML baselines
        'Logistic Regression': 'results_lr_top30.csv',
        'Random Forest': 'results_random_forest_top30.csv',
        'XGBoost': 'results_xgboost_top30.csv',
        # API experiments
        'Single-Agent 0-shot (Haiku)': 'results_api_single_0shot.csv',
        'Single-Agent 1-shot (Haiku)': 'results_api_single_1shot.csv',
        # Local experiments
        'Single-Agent 1-shot (Qwen-7B)': 'results_local_qwen7b_1shot.csv',
        # Modality agents
        'ACC Agent only': 'results_api_modality_acc.csv',
        'PPG Agent only': 'results_api_modality_ppg.csv',
        'Light Agent only': 'results_api_modality_light.csv',
        # Fusion methods
        'Statistical Fusion (majority vote)': 'results_api_modality_stat_fusion.csv',
        'Semantic Fusion': 'results_api_semantic_fusion.csv',
        'Hybrid Fusion (ConSensus)': 'results_api_hybrid_fusion.csv',
        # Local multi-agent — Qwen-7B
        'Hybrid Fusion Local (Qwen-7B)': 'results_local_multi_hybrid_fusion.csv',
        'Statistical Fusion Local (Qwen-7B)': 'results_local_multi_stat_fusion.csv',
        'Light Agent Local (Qwen-7B)': 'results_local_multi_light.csv',
        # Local multi-agent — Llama 3.2 3B
        'ACC Agent (Llama-3.2-3B)': 'results_llama_3_2_3b_instruct_multi_acc.csv',
        'PPG Agent (Llama-3.2-3B)': 'results_llama_3_2_3b_instruct_multi_ppg.csv',
        'Light Agent (Llama-3.2-3B)': 'results_llama_3_2_3b_instruct_multi_light.csv',
        'Statistical Fusion (Llama-3.2-3B)': 'results_llama_3_2_3b_instruct_multi_stat_fusion.csv',
        'Semantic Fusion (Llama-3.2-3B)': 'results_llama_3_2_3b_instruct_multi_semantic_fusion.csv',
        'Hybrid Fusion (Llama-3.2-3B)': 'results_llama_3_2_3b_instruct_multi_hybrid_fusion.csv',
        # Local multi-agent — OLMo 1B
        'ACC Agent (OLMo-1B)': 'results_olmo_2_0425_1b_instruct_multi_acc.csv',
        'PPG Agent (OLMo-1B)': 'results_olmo_2_0425_1b_instruct_multi_ppg.csv',
        'Light Agent (OLMo-1B)': 'results_olmo_2_0425_1b_instruct_multi_light.csv',
        'Statistical Fusion (OLMo-1B)': 'results_olmo_2_0425_1b_instruct_multi_stat_fusion.csv',
        'Semantic Fusion (OLMo-1B)': 'results_olmo_2_0425_1b_instruct_multi_semantic_fusion.csv',
        'Hybrid Fusion (OLMo-1B)': 'results_olmo_2_0425_1b_instruct_multi_hybrid_fusion.csv',
        # Deep Learning models
        'TCN': 'results_dl_tcn.csv',
        'LSTM': 'results_dl_lstm.csv',
        'Transformer': 'results_dl_transformer.csv',
    }

    # Column name mappings for different CSV formats
    col_map = {'bacc': 'balanced_accuracy', 'bal_acc': 'balanced_accuracy',
               'acc': 'accuracy', 'prec': 'precision', 'rec': 'recall'}

    rows = []
    per_subject = {}

    for method, fname in result_files.items():
        path = TEMP_EXPS / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.columns = [col_map.get(c, c) for c in df.columns]
        if 'balanced_accuracy' not in df.columns:
            continue

        ba = df['balanced_accuracy'].mean()
        ba_std = df['balanced_accuracy'].std()
        n = len(df)

        row = {'Method': method, 'Balanced Acc': ba, 'Std': ba_std, 'N_Subjects': n}
        for m in ['accuracy', 'f1', 'precision', 'recall', 'auc']:
            if m in df.columns:
                row[m.capitalize()] = df[m].mean()

        rows.append(row)
        per_subject[method] = df

    return pd.DataFrame(rows), per_subject


def export_modality_agent_examples(n=10):
    """Export example predictions with reasoning from modality agents."""
    outputs = load_pkl(TEMP_EXPS / 'modality_agent_raw_outputs.pkl')
    if outputs is None:
        return None

    df = pd.read_csv(TEMP_EXPS / 'modality_text_features.csv')
    df['category'] = df['category'].astype(int)

    examples = []
    # Pick samples with diverse predictions
    sample_indices = list(outputs.keys())[:n*3]
    np.random.seed(42)
    np.random.shuffle(sample_indices)

    count = 0
    for idx in sample_indices:
        if count >= n:
            break
        if idx not in df.index:
            continue
        row = df.loc[idx]
        out = outputs[idx]

        example = {
            'sample_index': int(idx),
            'subject': row['P_ID'],
            'true_label': 'interaction' if row['category'] == 1 else 'no_interaction',
            'acc_prediction': 'interaction' if out['acc_pred'] == 1 else 'no_interaction',
            'acc_reasoning': out.get('acc_reason', ''),
            'ppg_prediction': 'interaction' if out['ppg_pred'] == 1 else 'no_interaction',
            'ppg_reasoning': out.get('ppg_reason', ''),
            'light_prediction': 'interaction' if out['light_pred'] == 1 else 'no_interaction',
            'light_reasoning': out.get('light_reason', ''),
            'statistical_fusion': 'interaction' if out['stat_pred'] == 1 else 'no_interaction',
        }
        examples.append(example)
        count += 1

    return examples


def export_all_per_sample_predictions():
    """Combine all per-sample predictions into one CSV."""
    # Modality agent outputs
    mod_outputs = load_pkl(TEMP_EXPS / 'modality_agent_raw_outputs.pkl')
    sem_outputs = load_pkl(TEMP_EXPS / 'semantic_fusion_per_sample.pkl')
    hyb_outputs = load_pkl(TEMP_EXPS / 'hybrid_fusion_per_sample.pkl')

    if mod_outputs is None:
        return None

    df = pd.read_csv(TEMP_EXPS / 'modality_text_features.csv')
    df['category'] = df['category'].astype(int)

    rows = []
    for idx in mod_outputs:
        if idx not in df.index:
            continue
        row = df.loc[idx]
        out = mod_outputs[idx]
        r = {
            'index': idx,
            'P_ID': row['P_ID'],
            'true_label': row['category'],
            'acc_pred': out['acc_pred'],
            'acc_reason': out.get('acc_reason', ''),
            'ppg_pred': out['ppg_pred'],
            'ppg_reason': out.get('ppg_reason', ''),
            'light_pred': out['light_pred'],
            'light_reason': out.get('light_reason', ''),
            'stat_fusion_pred': out['stat_pred'],
        }
        if sem_outputs and idx in sem_outputs:
            r['semantic_fusion_pred'] = sem_outputs[idx][0]
            r['semantic_fusion_reason'] = sem_outputs[idx][1]
        if hyb_outputs and idx in hyb_outputs:
            r['hybrid_fusion_pred'] = hyb_outputs[idx][0]

        rows.append(r)

    return pd.DataFrame(rows)


def plot_results(summary_df, output_path):
    """Create bar chart comparing all methods."""
    if len(summary_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    methods = summary_df['Method'].values
    ba = summary_df['Balanced Acc'].values
    std = summary_df['Std'].values

    colors = []
    for m in methods:
        if 'Logistic' in m or 'Random' in m or 'XGBoost' in m:
            colors.append('#2196F3')  # blue for ML
        elif 'Single' in m:
            colors.append('#FF9800')  # orange for single-agent
        elif 'Agent only' in m:
            colors.append('#9C27B0')  # purple for modality agents
        elif 'Fusion' in m or 'ConSensus' in m:
            colors.append('#4CAF50')  # green for fusion
        else:
            colors.append('#607D8B')

    bars = ax.barh(range(len(methods)), ba, xerr=std, color=colors, alpha=0.8, capsize=3)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel('Balanced Accuracy (macro-averaged across LOSO folds)')
    ax.set_title('Social Interaction Detection: ML Baselines vs ConSensus-Style LLM Agents')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
    ax.legend()
    ax.set_xlim(0.3, 0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to {output_path}")


def plot_per_subject(per_subject, output_path):
    """Plot per-subject balanced accuracy for key methods."""
    key_methods = [
        'Logistic Regression',
        'Single-Agent 1-shot (Haiku)',
        'Statistical Fusion (majority vote)',
        'Hybrid Fusion (ConSensus)',
    ]
    available = {k: v for k, v in per_subject.items() if k in key_methods and 'balanced_accuracy' in v.columns}

    if len(available) < 2:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(38)
    width = 0.2
    colors = ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50']

    for i, (method, df) in enumerate(available.items()):
        subjects = sorted(df['subject'].unique()) if 'subject' in df.columns else []
        if len(subjects) > 0:
            ba = df.sort_values('subject')['balanced_accuracy'].values
            ax.bar(x[:len(ba)] + i * width, ba, width, label=method, color=colors[i % 4], alpha=0.7)

    ax.set_xlabel('Subject')
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Per-Subject Balanced Accuracy Comparison')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved per-subject plot to {output_path}")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / 'code').mkdir(exist_ok=True)
    (REPORT_DIR / 'results').mkdir(exist_ok=True)
    (REPORT_DIR / 'plots').mkdir(exist_ok=True)
    (REPORT_DIR / 'predictions').mkdir(exist_ok=True)
    (REPORT_DIR / 'logs').mkdir(exist_ok=True)

    print("=" * 60)
    print("  Compiling Comprehensive Report")
    print("=" * 60)

    # 1. Summary table
    print("\n--- Summary Table ---")
    summary_df, per_subject = compile_all_results()
    if len(summary_df) > 0:
        summary_df_sorted = summary_df.sort_values('Balanced Acc', ascending=False)
        print(summary_df_sorted.to_string(index=False))
        summary_df_sorted.to_csv(REPORT_DIR / 'results' / 'summary_all_methods.csv', index=False)

    # 2. Example predictions with reasoning
    print("\n--- Example Predictions ---")
    examples = export_modality_agent_examples(n=15)
    if examples:
        with open(REPORT_DIR / 'predictions' / 'example_predictions.json', 'w') as f:
            json.dump(examples, f, indent=2)
        print(f"  Saved {len(examples)} example predictions with reasoning")

        # Print a few for quick view
        for ex in examples[:3]:
            print(f"\n  Subject: {ex['subject']}, True: {ex['true_label']}")
            print(f"    ACC: {ex['acc_prediction']} — {ex['acc_reasoning'][:80]}")
            print(f"    PPG: {ex['ppg_prediction']} — {ex['ppg_reasoning'][:80]}")
            print(f"    Light: {ex['light_prediction']} — {ex['light_reasoning'][:80]}")
            print(f"    Majority Vote: {ex['statistical_fusion']}")

    # 3. All per-sample predictions CSV
    print("\n--- Per-Sample Predictions ---")
    per_sample_df = export_all_per_sample_predictions()
    if per_sample_df is not None:
        per_sample_df.to_csv(REPORT_DIR / 'predictions' / 'all_per_sample_predictions.csv', index=False)
        print(f"  Saved {len(per_sample_df)} per-sample predictions")

    # 4. Plots
    print("\n--- Generating Plots ---")
    if len(summary_df) > 0:
        plot_results(summary_df_sorted, REPORT_DIR / 'plots' / 'method_comparison.png')
        plot_per_subject(per_subject, REPORT_DIR / 'plots' / 'per_subject_comparison.png')

    # 5. Copy all code files
    print("\n--- Copying Code ---")
    import shutil
    for pattern in ['phase6_*.py', 'phase7_*.py', 'phase8_*.py']:
        for f in sorted(TEMP_EXPS.glob(pattern)):
            shutil.copy2(f, REPORT_DIR / 'code' / f.name)
            print(f"  {f.name}")

    # 6. Copy all result CSVs
    print("\n--- Copying Results ---")
    for pattern in ['results_*.csv', 'results_dl_*.csv']:
        for f in sorted(TEMP_EXPS.glob(pattern)):
            shutil.copy2(f, REPORT_DIR / 'results' / f.name)
            print(f"  {f.name}")

    # 7. Copy all per-sample prediction CSVs
    print("\n--- Copying Per-Sample Predictions ---")
    for pattern in ['*_per_sample.csv', 'dl_*_per_sample.csv']:
        for f in sorted(TEMP_EXPS.glob(pattern)):
            shutil.copy2(f, REPORT_DIR / 'predictions' / f.name)
            print(f"  {f.name}")

    # 8. Copy logs
    print("\n--- Copying Logs ---")
    for pattern in ['phase6_*.log', 'phase7_*.log', 'phase8_*.log']:
        for f in sorted(TEMP_EXPS.glob(pattern)):
            shutil.copy2(f, REPORT_DIR / 'logs' / f.name)
            print(f"  {f.name}")

    # 9. Copy pkl files
    for f in sorted(TEMP_EXPS.glob('*.pkl')):
        if 'features' not in f.name and 'raw_sensor' not in f.name:
            shutil.copy2(f, REPORT_DIR / 'predictions' / f.name)

    # 9. Copy paper
    paper = PROJECT_DIR / 'papers' / 'multimodal_agent.pdf'
    if paper.exists():
        shutil.copy2(paper, REPORT_DIR / 'ConSensus_paper.pdf')

    print(f"\n{'=' * 60}")
    print(f"  Report compiled at: {REPORT_DIR}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 60}")

    # Print directory tree
    print("\nDirectory structure:")
    for p in sorted(REPORT_DIR.rglob('*')):
        if p.is_file():
            depth = len(p.relative_to(REPORT_DIR).parts)
            indent = '  ' * depth
            size = p.stat().st_size
            size_str = f"{size/1024:.0f}KB" if size > 1024 else f"{size}B"
            print(f"{indent}{p.name} ({size_str})")


if __name__ == '__main__':
    main()
