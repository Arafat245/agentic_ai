"""
Phase 9 v2 — ReAct Agent with Lean Toolkit
Variant 1: LR + Light only (2 tools)
Variant 2: LR + RF + Light (3 tools)
Runs both variants sequentially. Saves all traces.
"""

import asyncio
import pandas as pd
import numpy as np
import json
import time
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

sys.path.insert(0, str(Path(__file__).parent))
from phase6_utils import (
    load_modality_texts, compute_metrics, RESULTS_DIR, SEED
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path('/mnt/sdb/arafat/agentic_ai/project')
CONCURRENCY = 5
SUBSAMPLE_PER_CLASS = 25
MAX_STEPS = 3
CONFIDENCE_THRESHOLD = 0.65

TOP_FEATURES = [
    'acc_x_band_high', 'acc_x_band_low', 'acc_z_band_high',
    'acc_x_spectral_centroid', 'acc_x_zcr', 'acc_y_min',
    'acc_y_band_high', 'acc_mag_band_high', 'acc_z_band_low',
    'acc_z_zcr', 'acc_z_spectral_centroid', 'acc_z_min',
    'acc_z_range', 'acc_x_spectral_entropy', 'acc_z_std',
    'acc_x_min', 'acc_y_spectral_centroid', 'acc_mag_min',
    'acc_y_zcr', 'acc_mag_range',
    'ppg_signal_quality_mean', 'ppg_hrv_pnn50',
    'ppg_signal_quality_std', 'ppg_hrv_pnn20',
    'ppg_zcr', 'ppg_peak_rate',
    'light_log_mean', 'light_kurtosis', 'light_log_std', 'light_n_changes',
]


class ToolKit:
    def __init__(self, train_features_df, train_y, use_rf=False, transformer_preds=None):
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(train_features_df[TOP_FEATURES].fillna(0).values)

        self.lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED, C=1.0)
        self.lr.fit(X_train, train_y)

        self.rf = None
        if use_rf:
            self.rf = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=SEED, n_jobs=4)
            self.rf.fit(X_train, train_y)

        self.transformer_preds = transformer_preds  # dict: idx -> (pred, prob)

    def _get_X(self, features_row):
        return self.scaler.transform(features_row[TOP_FEATURES].fillna(0).values.reshape(1, -1))

    def lr_predict(self, features_row):
        X = self._get_X(features_row)
        prob = self.lr.predict_proba(X)[0]
        label = "interaction" if prob[1] > 0.5 else "no_interaction"
        return f"LR predicts: {label} (probability: {max(prob):.3f}). no_interaction={prob[0]:.3f}, interaction={prob[1]:.3f}"

    def rf_predict(self, features_row):
        if self.rf is None:
            return "RF not available"
        X = self._get_X(features_row)
        prob = self.rf.predict_proba(X)[0]
        label = "interaction" if prob[1] > 0.5 else "no_interaction"
        return f"Random Forest predicts: {label} (probability: {max(prob):.3f}). no_interaction={prob[0]:.3f}, interaction={prob[1]:.3f}"

    def get_light_text(self, text_row):
        return text_row['text_light']

    def get_summary_stats(self, features_row):
        acc = features_row.get('acc_mag_band_high', 0)
        zcr = features_row.get('acc_x_zcr', 0)
        ppg_q = features_row.get('ppg_signal_quality_mean', 0)
        pnn50 = features_row.get('ppg_hrv_pnn50', 0)
        light_m = features_row.get('light_log_mean', 0)
        light_c = features_row.get('light_n_changes', 0)
        return (f"ACC: high-freq={acc:.3f}, ZCR={zcr:.3f} ({'active' if acc>0.15 else 'still'}); "
                f"PPG: quality={ppg_q:.3f}, pNN50={pnn50:.3f} ({'high HRV' if pnn50>0.8 else 'low HRV'}); "
                f"Light: log-mean={light_m:.2f}, changes={light_c:.0f} ({'dynamic' if light_c>30 else 'stable'})")

    def transformer_predict(self, idx):
        if self.transformer_preds is None or idx not in self.transformer_preds:
            return "Transformer prediction not available for this sample"
        pred, prob = self.transformer_preds[idx]
        label = "interaction" if pred == 1 else "no_interaction"
        return f"Transformer (DL) predicts: {label} (probability: {prob:.3f})"

    def execute(self, tool_name, features_row, text_row, idx=None):
        if tool_name == 'lr_predict':
            return self.lr_predict(features_row)
        elif tool_name == 'rf_predict':
            return self.rf_predict(features_row)
        elif tool_name == 'get_light_text':
            return self.get_light_text(text_row)
        elif tool_name == 'get_summary_stats':
            return self.get_summary_stats(features_row)
        elif tool_name == 'transformer_predict':
            return self.transformer_predict(idx)
        return f"Unknown tool: {tool_name}"


def get_system_prompt(variant):
    if variant == 'lr_light_summary':
        tools = """Available tools:
1. lr_predict - Logistic Regression prediction with probability (best ML model, 0.5653 balanced accuracy)
2. get_light_text - Ambient light sensor description (best LLM modality)
3. get_summary_stats - Quick summary of all 3 sensor modalities (ACC activity, PPG state, Light environment)"""
    elif variant == 'lr_rf':
        tools = """Available tools:
1. lr_predict - Logistic Regression prediction with probability (best ML model, 0.5653 balanced accuracy)
2. rf_predict - Random Forest prediction with probability (0.5411 balanced accuracy, different error pattern than LR)"""
    elif variant == 'lr_light':
        tools = """Available tools:
1. lr_predict - Logistic Regression prediction with probability (best ML model, 0.5653 balanced accuracy)
2. get_light_text - Ambient light sensor description (best LLM modality)"""
    else:  # lr_rf_trans_light
        tools = """Available tools:
1. lr_predict - Logistic Regression prediction with probability (best ML model, 0.5653 balanced accuracy)
2. rf_predict - Random Forest prediction with probability (0.5411 balanced accuracy, different error pattern than LR)
3. transformer_predict - Transformer (deep learning) prediction with probability (0.5479 balanced accuracy)
4. get_light_text - Ambient light sensor description (best LLM modality)"""

    return f"""You are an AI agent detecting social interactions from smartwatch sensor data.
You have access to tools that provide sensor analysis.

{tools}

STRATEGY:
1. ALWAYS call lr_predict first — it is the most accurate tool
2. If LR probability > 0.60, TRUST LR immediately — set it as your prediction
3. If LR probability is 0.50-0.60 (uncertain), call ONE more tool to verify, then decide
4. Only override LR if another tool STRONGLY disagrees
5. Maximum {MAX_STEPS} tool calls allowed

RESPOND in this exact JSON format:
{{"tool_call": "<tool_name>" or null, "reasoning": "<your reasoning>", "confidence": <0.0-1.0>, "prediction": "<interaction or no_interaction>" or null}}

- Set prediction to null until ready to decide
- On your LAST allowed step, you MUST set prediction
- Default to LR's prediction if unsure"""


async def run_react_sample(client, semaphore, toolkit, features_row, text_row, variant, sample_idx=None):
    system_prompt = get_system_prompt(variant)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Analyze this smartwatch sample. Is this a social interaction?"},
    ]
    trace = []

    for step in range(MAX_STEPS + 1):
        if step == MAX_STEPS:
            messages.append({"role": "user", "content": "LAST step. You MUST set prediction now."})

        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini", messages=messages, max_tokens=200, temperature=0,
                )
                reply = response.choices[0].message.content
        except Exception as e:
            if '429' in str(e):
                await asyncio.sleep(3)
            reply = json.dumps({"tool_call": None, "reasoning": f"API error", "confidence": 0.5, "prediction": "no_interaction"})

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
            pred_int = 0 if "no_interaction" in str(prediction) else (1 if "interaction" in str(prediction) else 0)
            return pred_int, confidence, trace

        if tool_call:
            tool_result = toolkit.execute(tool_call, features_row, text_row, idx=sample_idx)
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Tool result ({tool_call}): {tool_result}"})
        else:
            return 0, confidence, trace

    return 0, confidence, trace


async def run_variant(client, feat_df, text_df, variant, use_rf, trans_lookup=None):
    semaphore = asyncio.Semaphore(CONCURRENCY)
    subjects = sorted(feat_df['P_ID'].unique())
    all_results = []
    all_samples = []
    tag = f"react_{variant}"

    print(f"\n{'='*60}")
    print(f"  ReAct {variant} — LOSO-CV (38 subjects)")
    print(f"{'='*60}")

    t0 = time.time()
    total_calls = 0

    for fold_i, test_subj in enumerate(subjects):
        t_sub = time.time()
        train_mask = feat_df['P_ID'] != test_subj
        test_mask = feat_df['P_ID'] == test_subj

        test_feat = feat_df[test_mask]
        test_text = text_df[test_mask]
        rng = np.random.RandomState(SEED + fold_i)
        parts_f, parts_t = [], []
        for cat in [0, 1]:
            pool_idx = test_feat[test_feat['category'] == cat].index
            n = min(SUBSAMPLE_PER_CLASS, len(pool_idx))
            if n > 0:
                chosen = rng.choice(pool_idx, size=n, replace=False)
                parts_f.append(test_feat.loc[chosen])
                parts_t.append(test_text.loc[chosen])
        test_feat_sub = pd.concat(parts_f)
        test_text_sub = pd.concat(parts_t)

        toolkit = ToolKit(feat_df[train_mask], feat_df[train_mask]['category'].values, use_rf=use_rf, transformer_preds=trans_lookup)

        tasks = []
        indices = []
        for idx in test_feat_sub.index:
            tasks.append(run_react_sample(client, semaphore, toolkit, feat_df.loc[idx], text_df.loc[idx], variant, sample_idx=idx))
            indices.append(idx)

        results = await asyncio.gather(*tasks)

        y_true = test_feat_sub['category'].values
        y_pred = []
        for idx, (pred, conf, trace) in zip(indices, results):
            y_pred.append(pred)
            total_calls += len(trace)
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
        pd.DataFrame(all_results).to_csv(RESULTS_DIR / f'results_{tag}.csv', index=False)
        pd.DataFrame(all_samples).to_csv(RESULTS_DIR / f'{tag}_per_sample.csv', index=False)

    elapsed = time.time() - t0
    rdf = pd.DataFrame(all_results)
    ba = rdf['balanced_accuracy'].mean()
    print(f"\n  {variant}: BalAcc={ba:.4f}, {total_calls} API calls in {elapsed:.1f}s")
    print(f"  LR baseline: 0.5653")
    return ba


async def main():
    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    text_df = load_modality_texts()
    feat_df = pd.read_csv(RESULTS_DIR / 'all_subjects_features.csv')
    feat_df['category'] = feat_df['category'].astype(int)
    feat_df[TOP_FEATURES] = feat_df[TOP_FEATURES].fillna(0)
    text_df.index = feat_df.index

    # Load Transformer predictions as lookup
    trans_df = pd.read_csv(RESULTS_DIR / 'dl_transformer_per_sample.csv')
    trans_lookup = {}
    # Assign index matching feat_df order (same subject order, same sample order within subject)
    for subj in sorted(feat_df['P_ID'].unique()):
        subj_feat_idx = feat_df[feat_df['P_ID'] == subj].index
        subj_trans = trans_df[trans_df['P_ID'] == subj]
        for feat_idx, (_, trans_row) in zip(subj_feat_idx, subj_trans.iterrows()):
            trans_lookup[feat_idx] = (int(trans_row['pred']), float(trans_row['prob']))
    print(f"Loaded {len(trans_lookup)} Transformer predictions as lookup")

    # Variant 1: LR + Light + Summary (predicted best - drops bad ACC/PPG text)
    ba1 = await run_variant(client, feat_df, text_df, 'lr_light_summary', use_rf=False, trans_lookup=None)

    # Variant 2: LR + RF (ML tools only)
    ba2 = await run_variant(client, feat_df, text_df, 'lr_rf', use_rf=True, trans_lookup=None)

    # Variant 3: LR + RF + Transformer + Light (full best toolkit)
    ba3 = await run_variant(client, feat_df, text_df, 'lr_rf_trans_light', use_rf=True, trans_lookup=trans_lookup)

    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON (all GPT-4o-mini)")
    print(f"{'='*60}")
    print(f"  LR baseline:              0.5653")
    print(f"  Ensemble LR+RF+Trans:     0.5774")
    print(f"  ReAct Full (5 tools):      0.5632")
    print(f"  ReAct LR+Light+Summary:   {ba1:.4f}")
    print(f"  ReAct LR+RF:              {ba2:.4f}")
    print(f"  ReAct LR+RF+Trans+Light:  {ba3:.4f}")


if __name__ == '__main__':
    asyncio.run(main())
