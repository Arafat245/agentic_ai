"""
Phase 9 — ReAct Agent with Ensemble + Light tools (GPT-4o-mini)
Ensemble (LR+RF+Transformer) as primary tool + Light text for uncertain cases.
Starting from best ML baseline (0.5774), uses LLM to selectively override.
"""

import asyncio, pandas as pd, numpy as np, json, time, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env', override=True)

sys.path.insert(0, str(Path(__file__).parent))
from phase6_utils import load_modality_texts, compute_metrics, RESULTS_DIR, SEED

CONCURRENCY = 8
SUBSAMPLE_PER_CLASS = 25
MAX_STEPS = 2  # Only 2 steps needed: check ensemble, then predict (or check light)

SYSTEM_PROMPT = """You are an AI agent detecting social interactions from smartwatch sensor data.

Available tools:
1. ensemble_predict - Ensemble of LR + Random Forest + Transformer (our BEST ML baseline, 0.5774 balanced accuracy)
2. get_light_text - Ambient light sensor description

STRATEGY:
- Step 0: ALWAYS call ensemble_predict. Do NOT predict yet.
- Step 1: If ensemble probability > 0.60, TRUST it and predict. Otherwise call get_light_text.
- Step 2: MUST make final prediction.

Respond ONLY with JSON (no markdown):
{"tool_call": "name_or_null", "reasoning": "your reasoning", "confidence": 0.5, "prediction": null_or_label}"""


def build_ensemble_lookup():
    """Load saved ensemble predictions and build lookup by feature DataFrame index."""
    df = pd.read_csv(RESULTS_DIR / 'all_subjects_features.csv')
    df['category'] = df['category'].astype(int)
    ens = pd.read_csv(RESULTS_DIR / 'ensemble_lr_rf_trans_per_sample.csv')
    lookup = {}
    # Match by position within each subject (same order)
    for subj in sorted(df['P_ID'].unique()):
        subj_feat_idx = df[df['P_ID'] == subj].index.tolist()
        subj_ens = ens[ens['P_ID'] == subj]
        for feat_idx, (_, row) in zip(subj_feat_idx, subj_ens.iterrows()):
            lookup[feat_idx] = (int(row['pred']), float(row['prob']))
    return lookup, df


async def run_sample(client, semaphore, feat_row, text_row, idx, ensemble_lookup):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Analyze this smartwatch sample. Is this a social interaction?"}]
    trace = []

    for step in range(MAX_STEPS + 1):
        if step == MAX_STEPS:
            messages.append({"role": "user", "content": "LAST step. You MUST set prediction now."})

        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model="gpt-4o-mini", messages=messages, max_tokens=150, temperature=0,
                )
                reply = resp.choices[0].message.content
        except Exception as e:
            reply = json.dumps({"tool_call": None, "reasoning": "error", "confidence": 0.5, "prediction": "no_interaction"})

        try:
            parsed = json.loads(reply)
        except:
            parsed = {"tool_call": None, "reasoning": reply, "confidence": 0.5, "prediction": "no_interaction"}

        tool_call = parsed.get("tool_call")
        reasoning = parsed.get("reasoning", "")
        confidence = parsed.get("confidence", 0.5)
        prediction = parsed.get("prediction")

        trace.append({"step": step, "tool_call": tool_call, "reasoning": reasoning, "confidence": confidence, "prediction": prediction})

        if prediction is not None:
            pred = 0 if "no_interaction" in str(prediction) else (1 if "interaction" in str(prediction) else 0)
            return pred, confidence, trace

        if tool_call == "ensemble_predict":
            if idx in ensemble_lookup:
                p, prob = ensemble_lookup[idx]
                label = "interaction" if p == 1 else "no_interaction"
                result = f"Ensemble predicts: {label} (probability: {prob:.3f})"
            else:
                result = "Ensemble prediction not available"
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Tool result: {result}"})
        elif tool_call == "get_light_text":
            result = text_row['text_light']
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Tool result: {result}"})
        else:
            return 0, confidence, trace

    return 0, confidence, trace


async def main():
    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    text_df = load_modality_texts()
    ensemble_lookup, feat_df = build_ensemble_lookup()
    feat_df['category'] = feat_df['category'].astype(int)
    text_df.index = feat_df.index

    print(f"Loaded {len(ensemble_lookup)} ensemble predictions")

    subjects = sorted(feat_df['P_ID'].unique())
    semaphore = asyncio.Semaphore(CONCURRENCY)
    all_results = []
    all_samples = []

    print(f"{'='*60}")
    print(f"  ReAct Ensemble + Light (GPT-4o-mini) — LOSO-CV")
    print(f"{'='*60}")

    t0 = time.time()

    for fold_i, test_subj in enumerate(subjects):
        t_sub = time.time()
        test_feat = feat_df[feat_df['P_ID'] == test_subj]
        test_text = text_df[text_df['P_ID'] == test_subj]

        rng = np.random.RandomState(SEED + fold_i)
        parts_f, parts_t = [], []
        for cat in [0, 1]:
            pool = test_feat[test_feat['category'] == cat].index
            n = min(SUBSAMPLE_PER_CLASS, len(pool))
            if n > 0:
                chosen = rng.choice(pool, size=n, replace=False)
                parts_f.append(test_feat.loc[chosen])
                parts_t.append(test_text.loc[chosen])
        test_f = pd.concat(parts_f)
        test_t = pd.concat(parts_t)

        tasks = []
        indices = []
        for idx in test_f.index:
            tasks.append(run_sample(client, semaphore, feat_df.loc[idx], text_df.loc[idx], idx, ensemble_lookup))
            indices.append(idx)

        results = await asyncio.gather(*tasks)

        y_true = test_f['category'].values
        y_pred = []
        for idx, (pred, conf, trace) in zip(indices, results):
            y_pred.append(pred)
            all_samples.append({
                'P_ID': test_subj, 'true_label': int(feat_df.loc[idx, 'category']),
                'prediction': pred, 'confidence': conf, 'n_steps': len(trace),
                'trace': json.dumps(trace),
            })

        y_pred = np.array(y_pred)
        metrics = compute_metrics(y_true, y_pred)
        metrics['subject'] = test_subj
        metrics['n_test'] = len(y_true)
        metrics['n_pos'] = int(y_true.sum())
        metrics['avg_steps'] = np.mean([len(r[2]) for r in results])
        all_results.append(metrics)

        elapsed_sub = time.time() - t_sub
        print(f"  [{fold_i+1:2d}/38] {test_subj}: BalAcc={metrics['balanced_accuracy']:.3f}, Steps={metrics['avg_steps']:.1f} ({elapsed_sub:.1f}s)", flush=True)

        # Incremental save
        pd.DataFrame(all_results).to_csv(RESULTS_DIR / 'results_react_ensemble_light.csv', index=False)
        pd.DataFrame(all_samples).to_csv(RESULTS_DIR / 'react_ensemble_light_per_sample.csv', index=False)

    elapsed = time.time() - t0
    rdf = pd.DataFrame(all_results)
    ba = rdf['balanced_accuracy'].mean()
    print(f"\n  ReAct Ensemble+Light: BalAcc={ba:.4f}, Avg steps={rdf['avg_steps'].mean():.2f}, Time={elapsed:.1f}s")
    print(f"  Ensemble alone:       0.5774")
    print(f"  LR baseline:          0.5653")


if __name__ == '__main__':
    asyncio.run(main())
