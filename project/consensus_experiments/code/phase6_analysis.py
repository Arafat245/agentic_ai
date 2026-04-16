"""
Phase 6 — Analysis: Aggregate all results and update README.md
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_DIR = Path('/mnt/sdb/arafat/agentic_ai/project')
RESULTS_DIR = PROJECT_DIR / 'temp_exps'

# All expected result files and their display names
RESULT_FILES = {
    # ML baselines
    'results_lr_top30.csv': ('Logistic Regression', 'sklearn', 'balanced_accuracy'),
    'results_random_forest_top30.csv': ('Random Forest', 'sklearn', 'balanced_accuracy'),
    'results_xgboost_top30.csv': ('XGBoost', 'sklearn', 'balanced_accuracy'),
    # API single-agent
    'results_api_single_0shot.csv': ('Single-Agent 0-shot', 'API', 'balanced_accuracy'),
    'results_api_single_1shot.csv': ('Single-Agent 1-shot', 'API', 'balanced_accuracy'),
    # Local single-agent
    'results_local_qwen7b_1shot.csv': ('Single-Agent 1-shot', 'Qwen-7B', 'balanced_accuracy'),
    # API modality agents
    'results_api_modality_acc.csv': ('ACC Agent only', 'API', 'balanced_accuracy'),
    'results_api_modality_ppg.csv': ('PPG Agent only', 'API', 'balanced_accuracy'),
    'results_api_modality_light.csv': ('Light Agent only', 'API', 'balanced_accuracy'),
    # API fusion
    'results_api_modality_stat_fusion.csv': ('Statistical Fusion (majority)', 'API', 'balanced_accuracy'),
    'results_api_semantic_fusion.csv': ('Semantic Fusion', 'API', 'balanced_accuracy'),
    'results_api_hybrid_fusion.csv': ('Hybrid Fusion (ConSensus)', 'API', 'balanced_accuracy'),
    # Local multi-agent
    'results_local_multi_hybrid_fusion.csv': ('Hybrid Fusion (subsampled)', 'Qwen-7B', 'balanced_accuracy'),
    'results_local_multi_stat_fusion.csv': ('Statistical Fusion (subsampled)', 'Qwen-7B', 'balanced_accuracy'),
    'results_local_multi_semantic_fusion.csv': ('Semantic Fusion (subsampled)', 'Qwen-7B', 'balanced_accuracy'),
}

# Map old column names to standardized ones
METRIC_COL_MAP = {
    'bacc': 'balanced_accuracy',
    'bal_acc': 'balanced_accuracy',
    'acc': 'accuracy',
    'prec': 'precision',
    'rec': 'recall',
}


def load_result(filename):
    """Load a results CSV and return (macro_balanced_accuracy, per_subject_series)."""
    path = RESULTS_DIR / filename
    if not path.exists():
        return None, None

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [METRIC_COL_MAP.get(c, c) for c in df.columns]

    if 'balanced_accuracy' not in df.columns:
        return None, None

    macro_ba = df['balanced_accuracy'].mean()
    return macro_ba, df['balanced_accuracy']


def main():
    print("="*60)
    print("  Phase 6 Analysis — ConSensus Experiment Results")
    print("="*60)

    rows = []
    per_subject_data = {}

    for filename, (method_name, model_name, _) in RESULT_FILES.items():
        macro_ba, per_subject = load_result(filename)
        if macro_ba is not None:
            std_ba = per_subject.std()
            rows.append({
                'Method': method_name,
                'Model': model_name,
                'Balanced Acc': f"{macro_ba:.4f}",
                'Std': f"{std_ba:.4f}",
                'N Subjects': len(per_subject),
            })
            per_subject_data[f"{method_name} ({model_name})"] = per_subject.values
            print(f"  {method_name:35s} ({model_name:10s}): BalAcc={macro_ba:.4f} +/- {std_ba:.4f}")
        else:
            print(f"  {method_name:35s} ({model_name:10s}): NOT FOUND ({filename})")

    if not rows:
        print("\nNo results found!")
        return

    summary_df = pd.DataFrame(rows)
    summary_path = RESULTS_DIR / 'results_consensus_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")

    # Statistical significance test: compare best LLM method vs LR baseline
    lr_key = "Logistic Regression (sklearn)"
    if lr_key in per_subject_data:
        lr_data = per_subject_data[lr_key]
        print(f"\n{'='*60}")
        print("  Statistical Tests vs LR Baseline (paired t-test)")
        print(f"{'='*60}")
        for key, data in per_subject_data.items():
            if key == lr_key or len(data) != len(lr_data):
                continue
            t_stat, p_val = stats.ttest_rel(data, lr_data)
            diff = data.mean() - lr_data.mean()
            sig = "*" if p_val < 0.05 else ""
            print(f"  {key:45s}: diff={diff:+.4f}, p={p_val:.4f} {sig}")

    # Generate README update
    print(f"\n{'='*60}")
    print("  README.md Update (copy below)")
    print(f"{'='*60}")

    readme_section = "\n## LLM-Based Experiments (ConSensus-Style Multi-Agent)\n\n"
    readme_section += "Replication of the [ConSensus](https://arxiv.org/abs/2601.06453) multi-agent framework:\n"
    readme_section += "- **Modality Agents**: One LLM per sensor (ACC, PPG, Light), each produces prediction + reasoning\n"
    readme_section += "- **Statistical Fusion**: Majority vote of 3 modality agents\n"
    readme_section += "- **Semantic Fusion**: LLM aggregates cross-modal reasoning\n"
    readme_section += "- **Hybrid Fusion**: LLM arbitrates between semantic and statistical outputs\n"
    readme_section += "- **Evaluation**: LOSO-CV (38 subjects), 1-shot in-context learning\n\n"

    readme_section += "| Method | Model | Balanced Acc | Std |\n"
    readme_section += "|---|---|---|---|\n"
    for _, row in summary_df.iterrows():
        readme_section += f"| {row['Method']} | {row['Model']} | {row['Balanced Acc']} | {row['Std']} |\n"

    readme_section += "\n*All metrics are macro-averaged balanced accuracy across 38 LOSO-CV folds.*\n"

    print(readme_section)

    # Save the section for easy copy-paste
    section_path = RESULTS_DIR / 'readme_llm_section.md'
    with open(section_path, 'w') as f:
        f.write(readme_section)
    print(f"Saved README section to {section_path}")


if __name__ == '__main__':
    main()
