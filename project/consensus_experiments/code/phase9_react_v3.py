"""
Phase 9 v3 — ReAct Agent with LR+Light+Summary (3 tools) using Claude Haiku
Fixed prompt for Haiku JSON compliance.
"""

import asyncio
import pandas as pd
import numpy as np
import json
import re
import time
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

sys.path.insert(0, str(Path(__file__).parent))
from phase6_utils import load_modality_texts, compute_metrics, RESULTS_DIR, SEED
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CONCURRENCY = 10
SUBSAMPLE_PER_CLASS = 25
MAX_STEPS = 3

TOP_FEATURES = [
    'acc_x_band_high','acc_x_band_low','acc_z_band_high','acc_x_spectral_centroid','acc_x_zcr','acc_y_min',
    'acc_y_band_high','acc_mag_band_high','acc_z_band_low','acc_z_zcr','acc_z_spectral_centroid','acc_z_min',
    'acc_z_range','acc_x_spectral_entropy','acc_z_std','acc_x_min','acc_y_spectral_centroid','acc_mag_min',
    'acc_y_zcr','acc_mag_range','ppg_signal_quality_mean','ppg_hrv_pnn50','ppg_signal_quality_std',
    'ppg_hrv_pnn20','ppg_zcr','ppg_peak_rate','light_log_mean','light_kurtosis','light_log_std','light_n_changes',
]

SYSTEM_PROMPT = """You are an AI agent detecting social interactions from smartwatch sensor data.

Available tools:
1. lr_predict - Logistic Regression prediction with probability (best ML model, 0.5653 balanced accuracy)
2. get_light_text - Ambient light sensor description (best sensor for detecting social settings)
3. get_summary_stats - Quick statistical summary of all 3 sensor modalities

STRATEGY:
- ALWAYS call lr_predict first on step 0. Do NOT predict on step 0.
- If LR probability > 0.60, trust it — make your prediction on step 1.
- If LR is uncertain (0.50-0.60), call get_light_text or get_summary_stats to verify.
- Only override LR if another tool STRONGLY disagrees.
- Maximum 3 tool calls.

Respond ONLY with JSON (no markdown, no extra text):
{"tool_call": "name_or_null", "reasoning": "your reasoning", "confidence": 0.5, "prediction": null}

When ready to predict:
{"tool_call": null, "reasoning": "final reasoning", "confidence": 0.8, "prediction": "interaction_or_no_interaction"}"""


class ToolKit:
    def __init__(self, train_df, train_y):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(train_df[TOP_FEATURES].fillna(0).values)
        self.lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED, C=1.0)
        self.lr.fit(X, train_y)

    def lr_predict(self, row):
        X = self.scaler.transform(row[TOP_FEATURES].fillna(0).values.reshape(1, -1))
        prob = self.lr.predict_proba(X)[0]
        label = "interaction" if prob[1] > 0.5 else "no_interaction"
        return f"LR predicts: {label} (probability: {max(prob):.3f}). no_interaction={prob[0]:.3f}, interaction={prob[1]:.3f}"

    def get_light_text(self, text_row):
        return text_row['text_light']

    def get_summary_stats(self, row):
        acc = row.get('acc_mag_band_high', 0)
        zcr = row.get('acc_x_zcr', 0)
        ppg_q = row.get('ppg_signal_quality_mean', 0)
        pnn50 = row.get('ppg_hrv_pnn50', 0)
        light_m = row.get('light_log_mean', 0)
        light_c = row.get('light_n_changes', 0)
        return (f"ACC: high-freq={acc:.3f}, ZCR={zcr:.3f} ({'active' if acc>0.15 else 'still'}); "
                f"PPG: quality={ppg_q:.3f}, pNN50={pnn50:.3f} ({'high HRV' if pnn50>0.8 else 'low HRV'}); "
                f"Light: log-mean={light_m:.2f}, changes={light_c:.0f} ({'dynamic' if light_c>30 else 'stable'})")

    def execute(self, tool_name, feat_row, text_row):
        if tool_name == 'lr_predict': return self.lr_predict(feat_row)
        elif tool_name == 'get_light_text': return self.get_light_text(text_row)
        elif tool_name == 'get_summary_stats': return self.get_summary_stats(feat_row)
        return f"Unknown tool: {tool_name}"


def parse_json(text):
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try: return json.loads(match.group())
        except: pass
    try: return json.loads(text)
    except: return {"tool_call": None, "reasoning": text, "confidence": 0.5, "prediction": "no_interaction"}


async def run_sample(client, semaphore, toolkit, feat_row, text_row):
    api_msgs = [{"role": "user", "content": "Analyze this smartwatch sample. Is this a social interaction?"}]
    trace = []

    for step in range(MAX_STEPS + 1):
        if step == MAX_STEPS:
            api_msgs.append({"role": "user", "content": "LAST step. You MUST set prediction now."})

        try:
            async with semaphore:
                resp = await client.messages.create(
                    model="claude-haiku-4-5-20251001", system=SYSTEM_PROMPT,
                    messages=api_msgs, max_tokens=150,
                )
                reply = resp.content[0].text
        except Exception as e:
            reply = json.dumps({"tool_call": None, "reasoning": f"error", "confidence": 0.5, "prediction": "no_interaction"})

        parsed = parse_json(reply)
        tool_call = parsed.get("tool_call")
        reasoning = parsed.get("reasoning", "")
        confidence = parsed.get("confidence", 0.5)
        prediction = parsed.get("prediction")

        trace.append({"step": step, "tool_call": tool_call, "reasoning": reasoning, "confidence": confidence, "prediction": prediction})

        if prediction is not None:
            pred = 0 if "no_interaction" in str(prediction) else (1 if "interaction" in str(prediction) else 0)
            return pred, confidence, trace

        if tool_call:
            result = toolkit.execute(tool_call, feat_row, text_row)
            api_msgs.append({"role": "assistant", "content": reply})
            api_msgs.append({"role": "user", "content": f"Tool result ({tool_call}): {result}"})
        else:
            return 0, confidence, trace

    return 0, confidence, trace


async def main():
    import anthropic
    client = anthropic.AsyncAnthropic()

    text_df = load_modality_texts()
    feat_df = pd.read_csv(RESULTS_DIR / 'all_subjects_features.csv')
    feat_df['category'] = feat_df['category'].astype(int)
    feat_df[TOP_FEATURES] = feat_df[TOP_FEATURES].fillna(0)
    text_df.index = feat_df.index

    subjects = sorted(feat_df['P_ID'].unique())
    semaphore = asyncio.Semaphore(CONCURRENCY)
    all_results = []
    all_samples = []

    print(f"{'='*60}")
    print(f"  ReAct LR+Light+Summary (Haiku) — LOSO-CV")
    print(f"{'='*60}")

    t0 = time.time()

    for fold_i, test_subj in enumerate(subjects):
        t_sub = time.time()
        train_mask = feat_df['P_ID'] != test_subj
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

        toolkit = ToolKit(feat_df[train_mask], feat_df[train_mask]['category'].values)

        tasks = []
        indices = []
        for idx in test_f.index:
            tasks.append(run_sample(client, semaphore, toolkit, feat_df.loc[idx], text_df.loc[idx]))
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
        pd.DataFrame(all_results).to_csv(RESULTS_DIR / 'results_react_lr_light_summary_haiku.csv', index=False)
        pd.DataFrame(all_samples).to_csv(RESULTS_DIR / 'react_lr_light_summary_haiku_per_sample.csv', index=False)

    elapsed = time.time() - t0
    rdf = pd.DataFrame(all_results)
    ba = rdf['balanced_accuracy'].mean()
    print(f"\n  Final: BalAcc={ba:.4f}, Avg steps={rdf['avg_steps'].mean():.2f}, Time={elapsed:.1f}s")
    print(f"  LR baseline: 0.5653")
    print(f"  ReAct Full (GPT-4o-mini): 0.5632")


if __name__ == '__main__':
    asyncio.run(main())
