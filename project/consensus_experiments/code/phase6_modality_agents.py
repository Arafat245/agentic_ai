"""
Phase 6 — Multi-Agent Modality Agents via API
3 modality-specific agents (ACC, PPG, Light) + statistical fusion (majority vote).
Full LOSO-CV on 33,727 samples. Each sample = 3 parallel API calls.
"""

import asyncio
import pandas as pd
import numpy as np
import pickle
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phase6_utils import (
    load_modality_texts, get_loso_splits, get_few_shot_examples,
    subsample_test, parse_prediction, compute_metrics, save_subject_results,
    build_messages_modality, statistical_fusion,
    get_api_client, RESULTS_DIR, SEED
)

CONCURRENCY = 20  # Anthropic has generous per-minute limits
SUBSAMPLE_PER_CLASS = 50  # 50 per class = ~100 per subject
MODALITIES = ['acc', 'ppg', 'light']


async def classify_one_sample(row, call_fn, client, semaphore, few_shots):
    """Run 3 modality agents in parallel for one sample."""
    tasks = []
    for mod in MODALITIES:
        messages = build_messages_modality(mod, row[f'text_{mod}'], few_shots[mod])
        tasks.append(call_fn(client, messages, semaphore))

    responses = await asyncio.gather(*tasks)

    result = {}
    preds = []
    for mod, resp in zip(MODALITIES, responses):
        pred, reason = parse_prediction(resp)
        result[f'{mod}_pred'] = pred
        result[f'{mod}_reason'] = reason
        result[f'{mod}_raw'] = resp
        preds.append(pred)

    # Statistical fusion (majority vote)
    result['stat_pred'] = statistical_fusion(*preds)
    return result


async def run_modality_agents_loso(df, call_fn, client, provider_name):
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Collect per-sample outputs for fusion agents later
    all_outputs = {}  # keyed by (P_ID, index)

    # Per-modality and statistical fusion results
    results_by_method = {mod: [] for mod in MODALITIES}
    results_by_method['stat_fusion'] = []

    print(f"\n{'='*60}")
    print(f"  Modality Agents 1-shot — {provider_name} — LOSO-CV")
    print(f"{'='*60}")

    t0 = time.time()
    total_calls = 0

    for fold_i, (test_subj, train_df, test_df) in enumerate(get_loso_splits(df)):
        t_sub = time.time()

        # Subsample test set
        test_df = subsample_test(test_df, n_per_class=SUBSAMPLE_PER_CLASS, seed=SEED + fold_i)

        # Get modality-specific few-shot examples
        few_shots = {}
        for mod in MODALITIES:
            few_shots[mod] = get_few_shot_examples(train_df, 1, modality=mod, seed=SEED + fold_i)

        # Process all test samples for this subject
        tasks = []
        indices = []
        for idx, row in test_df.iterrows():
            tasks.append(classify_one_sample(row, call_fn, client, semaphore, few_shots))
            indices.append(idx)

        sample_results = await asyncio.gather(*tasks)
        total_calls += len(tasks) * 3

        # Collect outputs
        y_true = test_df['category'].values

        for mod in MODALITIES:
            y_pred_mod = np.array([r[f'{mod}_pred'] for r in sample_results])
            metrics = compute_metrics(y_true, y_pred_mod)
            metrics['subject'] = test_subj
            metrics['n_test'] = len(y_true)
            metrics['n_pos'] = int(y_true.sum())
            results_by_method[mod].append(metrics)

        y_pred_stat = np.array([r['stat_pred'] for r in sample_results])
        metrics_stat = compute_metrics(y_true, y_pred_stat)
        metrics_stat['subject'] = test_subj
        metrics_stat['n_test'] = len(y_true)
        metrics_stat['n_pos'] = int(y_true.sum())
        results_by_method['stat_fusion'].append(metrics_stat)

        # Store raw outputs for fusion agents
        for idx_pos, (idx, r) in enumerate(zip(indices, sample_results)):
            all_outputs[idx] = r

        elapsed_sub = time.time() - t_sub
        ba_acc = results_by_method['acc'][-1]['balanced_accuracy']
        ba_stat = results_by_method['stat_fusion'][-1]['balanced_accuracy']
        print(f"  [{fold_i+1:2d}/38] {test_subj}: "
              f"ACC={ba_acc:.3f}, Stat={ba_stat:.3f} "
              f"({len(y_true)} samples, {elapsed_sub:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  Total: {total_calls} API calls in {elapsed:.1f}s")

    # Save per-modality results
    macros = {}
    for method, results in results_by_method.items():
        tag = f"api_modality_{method}"
        output_path = RESULTS_DIR / f'results_{tag}.csv'
        print(f"\n  --- {method.upper()} ---")
        macro = save_subject_results(results, output_path)
        macros[method] = macro

    # Save raw outputs for fusion agents
    outputs_path = RESULTS_DIR / 'modality_agent_raw_outputs.pkl'
    with open(outputs_path, 'wb') as f:
        pickle.dump(all_outputs, f)
    print(f"\n  Saved raw outputs to {outputs_path}")

    return macros, all_outputs


async def main():
    df = load_modality_texts()
    print(f"Loaded {len(df)} samples, {df['P_ID'].nunique()} subjects")

    call_fn, client, provider_name = get_api_client()

    macros, _ = await run_modality_agents_loso(df, call_fn, client, provider_name)

    print(f"\n{'='*60}")
    print("  MODALITY AGENTS SUMMARY (Balanced Accuracy)")
    print(f"{'='*60}")
    for method, macro in macros.items():
        print(f"  {method:15s}: {macro['balanced_accuracy']:.4f}")
    print(f"  {'ML Baseline':15s}: 0.5761")


if __name__ == '__main__':
    asyncio.run(main())
