"""
Phase 6 — Single-Agent API Baseline (0-shot + 1-shot)
Sends all modalities combined to one LLM agent.
Full LOSO-CV on 33,727 samples. Auto-detects Anthropic or OpenAI API.
"""

import asyncio
import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phase6_utils import (
    load_modality_texts, get_loso_splits, get_few_shot_examples,
    subsample_test, parse_prediction, compute_metrics, save_subject_results,
    build_messages_single, get_api_client, RESULTS_DIR, SEED
)

CONCURRENCY = 20  # Anthropic has generous per-minute limits
SUBSAMPLE_PER_CLASS = 50  # 50 per class = ~100 per subject, ~3800 total


async def run_single_agent_loso(df, call_fn, client, shot_count, provider_name):
    semaphore = asyncio.Semaphore(CONCURRENCY)
    all_results = []

    print(f"\n{'='*60}")
    print(f"  Single-Agent {shot_count}-shot — {provider_name} — LOSO-CV")
    print(f"{'='*60}")

    t0 = time.time()
    total_calls = 0
    total_parse_errors = 0

    for fold_i, (test_subj, train_df, test_df) in enumerate(get_loso_splits(df)):
        t_sub = time.time()

        # Subsample test set
        test_df = subsample_test(test_df, n_per_class=SUBSAMPLE_PER_CLASS, seed=SEED + fold_i)

        # Get few-shot examples
        few_shot = get_few_shot_examples(train_df, shot_count, modality='all', seed=SEED + fold_i)

        # Build all messages for this subject's test samples
        tasks = []
        for _, row in test_df.iterrows():
            messages = build_messages_single(row['text_all'], few_shot)
            tasks.append(call_fn(client, messages, semaphore))

        # Run all API calls for this subject concurrently
        responses = await asyncio.gather(*tasks)

        y_true = test_df['category'].values
        y_pred = []
        for resp in responses:
            pred, reason = parse_prediction(resp)
            y_pred.append(pred)
            if 'ERROR' in str(resp):
                total_parse_errors += 1

        y_pred = np.array(y_pred)
        total_calls += len(tasks)

        metrics = compute_metrics(y_true, y_pred)
        metrics['subject'] = test_subj
        metrics['n_test'] = len(y_true)
        metrics['n_pos'] = int(y_true.sum())
        all_results.append(metrics)

        elapsed_sub = time.time() - t_sub
        print(f"  [{fold_i+1:2d}/38] {test_subj}: "
              f"BalAcc={metrics['balanced_accuracy']:.3f}, F1={metrics['f1']:.3f} "
              f"({metrics['n_test']} samples, {elapsed_sub:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  Total: {total_calls} API calls in {elapsed:.1f}s ({total_parse_errors} parse errors)")

    # Save results
    tag = f"api_single_{shot_count}shot"
    output_path = RESULTS_DIR / f'results_{tag}.csv'
    macro = save_subject_results(all_results, output_path)

    return all_results, macro


async def main():
    df = load_modality_texts()
    print(f"Loaded {len(df)} samples, {df['P_ID'].nunique()} subjects")
    print(f"Class distribution: {dict(df['category'].value_counts().sort_index())}")

    call_fn, client, provider_name = get_api_client()

    # Run 0-shot
    _, macro_0shot = await run_single_agent_loso(df, call_fn, client, 0, provider_name)

    # Run 1-shot
    _, macro_1shot = await run_single_agent_loso(df, call_fn, client, 1, provider_name)

    # Summary
    print(f"\n{'='*60}")
    print("  SINGLE-AGENT SUMMARY")
    print(f"{'='*60}")
    print(f"  0-shot Balanced Accuracy: {macro_0shot['balanced_accuracy']:.4f}")
    print(f"  1-shot Balanced Accuracy: {macro_1shot['balanced_accuracy']:.4f}")
    print(f"  ML Baseline (LR):         0.5761")


if __name__ == '__main__':
    asyncio.run(main())
