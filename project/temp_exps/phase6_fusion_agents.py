"""
Phase 6 — Fusion Agents via API (Semantic + Hybrid)
Reads modality agent outputs from Phase 3 and runs fusion.
Full LOSO-CV on 33,727 samples.
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
    load_modality_texts, get_loso_splits,
    parse_prediction, compute_metrics, save_subject_results,
    build_messages_semantic_fusion, build_messages_hybrid_fusion,
    statistical_fusion, get_api_client, RESULTS_DIR, SEED
)

CONCURRENCY = 20  # Anthropic has generous per-minute limits


async def run_semantic_fusion(df, all_outputs, call_fn, client, provider_name):
    """Run semantic fusion agent on all samples."""
    semaphore = asyncio.Semaphore(CONCURRENCY)
    semantic_results_per_subject = []
    semantic_outputs = {}  # idx -> (pred, reason)

    print(f"\n{'='*60}")
    print(f"  Semantic Fusion — {provider_name} — LOSO-CV")
    print(f"{'='*60}")

    t0 = time.time()

    for fold_i, (test_subj, train_df, test_df) in enumerate(get_loso_splits(df)):
        t_sub = time.time()
        tasks = []
        indices = []

        # Only process samples that exist in modality agent outputs (subsampled)
        test_sub = test_df[test_df.index.isin(all_outputs.keys())]
        for idx, row in test_sub.iterrows():
            out = all_outputs[idx]
            acc_label = "interaction" if out['acc_pred'] == 1 else "no_interaction"
            ppg_label = "interaction" if out['ppg_pred'] == 1 else "no_interaction"
            light_label = "interaction" if out['light_pred'] == 1 else "no_interaction"

            messages = build_messages_semantic_fusion(
                acc_label, out['acc_reason'],
                ppg_label, out['ppg_reason'],
                light_label, out['light_reason'],
            )
            tasks.append(call_fn(client, messages, semaphore))
            indices.append(idx)

        responses = await asyncio.gather(*tasks)

        y_true = test_sub['category'].values
        y_pred = []
        for idx, resp in zip(indices, responses):
            pred, reason = parse_prediction(resp)
            y_pred.append(pred)
            semantic_outputs[idx] = (pred, reason)

        y_pred = np.array(y_pred)
        metrics = compute_metrics(y_true, y_pred)
        metrics['subject'] = test_subj
        metrics['n_test'] = len(y_true)
        metrics['n_pos'] = int(y_true.sum())
        semantic_results_per_subject.append(metrics)

        elapsed_sub = time.time() - t_sub
        print(f"  [{fold_i+1:2d}/38] {test_subj}: "
              f"BalAcc={metrics['balanced_accuracy']:.3f} "
              f"({metrics['n_test']} samples, {elapsed_sub:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  Total: {len(semantic_outputs)} calls in {elapsed:.1f}s")

    output_path = RESULTS_DIR / 'results_api_semantic_fusion.csv'
    macro = save_subject_results(semantic_results_per_subject, output_path)

    # Save per-sample semantic predictions
    with open(RESULTS_DIR / 'semantic_fusion_per_sample.pkl', 'wb') as f:
        pickle.dump(semantic_outputs, f)
    print(f"  Saved per-sample semantic predictions")

    return macro, semantic_outputs


async def run_hybrid_fusion(df, all_outputs, semantic_outputs, call_fn, client, provider_name):
    """Run hybrid fusion agent that combines semantic + statistical."""
    semaphore = asyncio.Semaphore(CONCURRENCY)
    hybrid_results_per_subject = []
    hybrid_per_sample = {}  # per-sample predictions

    print(f"\n{'='*60}")
    print(f"  Hybrid Fusion — {provider_name} — LOSO-CV")
    print(f"{'='*60}")

    t0 = time.time()

    for fold_i, (test_subj, train_df, test_df) in enumerate(get_loso_splits(df)):
        t_sub = time.time()
        tasks = []
        indices = []

        # Only process samples that exist in modality agent outputs (subsampled)
        test_sub = test_df[test_df.index.isin(all_outputs.keys())]
        for idx, row in test_sub.iterrows():
            out = all_outputs[idx]
            sem_pred, sem_reason = semantic_outputs[idx]

            # Labels for messages
            sem_label = "interaction" if sem_pred == 1 else "no_interaction"
            stat_pred = statistical_fusion(out['acc_pred'], out['ppg_pred'], out['light_pred'])
            stat_label = "interaction" if stat_pred == 1 else "no_interaction"
            acc_label = "interaction" if out['acc_pred'] == 1 else "no_interaction"
            ppg_label = "interaction" if out['ppg_pred'] == 1 else "no_interaction"
            light_label = "interaction" if out['light_pred'] == 1 else "no_interaction"

            messages = build_messages_hybrid_fusion(
                sem_label, sem_reason, stat_label,
                acc_label, ppg_label, light_label,
            )
            tasks.append(call_fn(client, messages, semaphore))
            indices.append(idx)

        responses = await asyncio.gather(*tasks)

        y_true = test_sub['category'].values
        y_pred = []
        for idx, resp in zip(indices, responses):
            pred, reason = parse_prediction(resp)
            y_pred.append(pred)
            hybrid_per_sample[idx] = (pred, reason)

        y_pred = np.array(y_pred)
        metrics = compute_metrics(y_true, y_pred)
        metrics['subject'] = test_subj
        metrics['n_test'] = len(y_true)
        metrics['n_pos'] = int(y_true.sum())
        hybrid_results_per_subject.append(metrics)

        elapsed_sub = time.time() - t_sub
        print(f"  [{fold_i+1:2d}/38] {test_subj}: "
              f"BalAcc={metrics['balanced_accuracy']:.3f} "
              f"({metrics['n_test']} samples, {elapsed_sub:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  Total: {len(df)} calls in {elapsed:.1f}s")

    output_path = RESULTS_DIR / 'results_api_hybrid_fusion.csv'
    macro = save_subject_results(hybrid_results_per_subject, output_path)

    # Save per-sample hybrid predictions
    with open(RESULTS_DIR / 'hybrid_fusion_per_sample.pkl', 'wb') as f:
        pickle.dump(hybrid_per_sample, f)
    print(f"  Saved per-sample hybrid predictions")

    return macro


async def main():
    df = load_modality_texts()
    print(f"Loaded {len(df)} samples, {df['P_ID'].nunique()} subjects")

    # Load modality agent outputs from Phase 3
    outputs_path = RESULTS_DIR / 'modality_agent_raw_outputs.pkl'
    if not outputs_path.exists():
        print(f"ERROR: {outputs_path} not found. Run phase6_modality_agents.py first.")
        return

    with open(outputs_path, 'rb') as f:
        all_outputs = pickle.load(f)
    print(f"Loaded {len(all_outputs)} modality agent outputs")

    call_fn, client, provider_name = get_api_client()

    # Run semantic fusion
    macro_sem, semantic_outputs = await run_semantic_fusion(df, all_outputs, call_fn, client, provider_name)

    # Run hybrid fusion
    macro_hyb = await run_hybrid_fusion(df, all_outputs, semantic_outputs, call_fn, client, provider_name)

    # Summary
    print(f"\n{'='*60}")
    print("  FUSION SUMMARY (Balanced Accuracy)")
    print(f"{'='*60}")
    print(f"  Semantic Fusion:  {macro_sem['balanced_accuracy']:.4f}")
    print(f"  Hybrid Fusion:    {macro_hyb['balanced_accuracy']:.4f}")
    print(f"  ML Baseline (LR): 0.5761")


if __name__ == '__main__':
    asyncio.run(main())
