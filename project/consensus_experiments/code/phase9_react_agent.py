"""
Phase 9 — ReAct Agent with Full Toolkit for Social Interaction Detection
Tools: LR predict, Transformer predict, MOMENT kNN, sensor text
Uses GPT-4o-mini as the reasoning agent.
Confidence-based loop: agent keeps calling tools until confident.
LOSO-CV with subsampled test (50/subject) for fair comparison.
Saves all per-sample predictions, reasoning traces, and tool call logs.
"""

import asyncio
import pandas as pd
import numpy as np
import pickle
import json
import time
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

sys.path.insert(0, str(Path(__file__).parent))
from phase6_utils import (
    load_modality_texts, get_loso_splits, subsample_test,
    compute_metrics, save_subject_results, RESULTS_DIR, SEED
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

PROJECT_DIR = Path('/mnt/sdb/arafat/agentic_ai/project')
CONCURRENCY = 5  # Conservative for OpenAI
SUBSAMPLE_PER_CLASS = 25
MAX_STEPS = 3  # Max tool calls per sample
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


# ===================== TOOL IMPLEMENTATIONS =====================

class ToolKit:
    """All tools the ReAct agent can call. Initialized per LOSO fold."""

    def __init__(self, train_features_df, train_y):
        """Pre-train LR and Transformer on training data for this fold."""
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(train_features_df[TOP_FEATURES].fillna(0).values)
        self.train_y = train_y

        # Train LR
        self.lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED, C=1.0)
        self.lr.fit(X_train, train_y)

    def lr_predict(self, features_row):
        """Tool: Get LR prediction + probability."""
        X = self.scaler.transform(features_row[TOP_FEATURES].fillna(0).values.reshape(1, -1))
        pred = self.lr.predict(X)[0]
        prob = self.lr.predict_proba(X)[0]
        label = "interaction" if pred == 1 else "no_interaction"
        conf = max(prob)
        return f"LR predicts: {label} (probability: {conf:.3f}). Class probabilities: no_interaction={prob[0]:.3f}, interaction={prob[1]:.3f}"

    def get_sensor_text(self, modality, text_row):
        """Tool: Get natural language description of sensor features."""
        col = f'text_{modality}'
        return text_row[col]

    def get_summary_stats(self, features_row):
        """Tool: Quick statistical summary across all modalities."""
        acc_energy = features_row.get('acc_mag_band_high', 0)
        acc_zcr = features_row.get('acc_x_zcr', 0)
        ppg_quality = features_row.get('ppg_signal_quality_mean', 0)
        ppg_pnn50 = features_row.get('ppg_hrv_pnn50', 0)
        light_mean = features_row.get('light_log_mean', 0)
        light_changes = features_row.get('light_n_changes', 0)

        parts = []
        parts.append(f"ACC: high-freq power={acc_energy:.3f}, ZCR={acc_zcr:.3f} ({'active' if acc_energy > 0.15 else 'still'})")
        parts.append(f"PPG: quality={ppg_quality:.3f}, pNN50={ppg_pnn50:.3f} ({'high HRV' if ppg_pnn50 > 0.8 else 'low HRV'})")
        parts.append(f"Light: log-mean={light_mean:.2f}, changes={light_changes:.0f} ({'dynamic' if light_changes > 30 else 'stable'})")
        return "; ".join(parts)


TOOL_DESCRIPTIONS = """Available tools:
1. lr_predict - Get Logistic Regression prediction with probability. Best overall model (0.5653 balanced accuracy).
2. get_sensor_text(acc) - Get detailed accelerometer feature description
3. get_sensor_text(ppg) - Get detailed PPG/heart rate feature description
4. get_sensor_text(light) - Get detailed ambient light feature description
5. get_summary_stats - Get quick statistical summary of all three sensor modalities"""

SYSTEM_PROMPT = f"""You are an AI agent detecting social interactions from smartwatch sensor data.
You have access to tools that provide sensor analysis. Use them to build evidence before making a prediction.

{TOOL_DESCRIPTIONS}

STRATEGY:
1. ALWAYS start by calling lr_predict — it is the most accurate tool (0.5653 balanced accuracy)
2. If LR probability > 0.60, TRUST LR's prediction immediately — set it as your prediction
3. If LR probability is 0.50-0.60 (uncertain), call ONE more tool to verify, then decide
4. Only override LR if you have STRONG evidence from sensors that contradicts it
5. Maximum {MAX_STEPS} tool calls allowed

RESPOND in this exact JSON format:
{{"tool_call": "<tool_name>" or null, "tool_arg": "<argument>" or null, "reasoning": "<your reasoning>", "confidence": <0.0-1.0>, "prediction": "<interaction or no_interaction>" or null}}

- Set prediction to null until you are ready to decide
- When you set prediction, that is your final answer
- On your LAST allowed step, you MUST set prediction (cannot be null)
- Default to LR's prediction if unsure"""


# ===================== AGENT LOOP =====================

async def run_react_agent(client, semaphore, toolkit, features_row, text_row):
    """Run the ReAct loop for one sample. Returns (prediction, trace)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": "Analyze this smartwatch sample and determine if a social interaction occurred."})

    trace = []  # Log all steps

    for step in range(MAX_STEPS + 1):
        # On last step, tell agent it must decide
        if step == MAX_STEPS:
            messages.append({"role": "user", "content": "This is your LAST step. You MUST set prediction now based on all evidence gathered. Do NOT set prediction to null."})

        # Call LLM
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=200,
                    temperature=0,
                )
                reply = response.choices[0].message.content
        except Exception as e:
            reply = json.dumps({"tool_call": None, "reasoning": f"API error: {e}", "confidence": 0.5, "prediction": "no_interaction"})

        # Parse response
        try:
            parsed = json.loads(reply)
        except json.JSONDecodeError:
            # Try to extract from non-JSON response
            parsed = {"tool_call": None, "reasoning": reply, "confidence": 0.5, "prediction": "no_interaction"}

        tool_call = parsed.get("tool_call")
        tool_arg = parsed.get("tool_arg")
        reasoning = parsed.get("reasoning", "")
        confidence = parsed.get("confidence", 0.5)
        prediction = parsed.get("prediction")

        trace.append({
            "step": step,
            "tool_call": tool_call,
            "tool_arg": tool_arg,
            "reasoning": reasoning,
            "confidence": confidence,
            "prediction": prediction,
        })

        # If prediction is made, we're done
        if prediction is not None:
            pred_int = 1 if "interaction" in str(prediction) and "no_interaction" not in str(prediction) else 0
            if "no_interaction" in str(prediction):
                pred_int = 0
            return pred_int, confidence, trace

        # Execute tool call
        if tool_call:
            if tool_call == "lr_predict":
                tool_result = toolkit.lr_predict(features_row)
            elif tool_call == "get_sensor_text" and tool_arg in ['acc', 'ppg', 'light']:
                tool_result = toolkit.get_sensor_text(tool_arg, text_row)
            elif tool_call == "get_summary_stats":
                tool_result = toolkit.get_summary_stats(features_row)
            else:
                tool_result = f"Unknown tool: {tool_call}"

            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Tool result ({tool_call}): {tool_result}"})
        else:
            # No tool call and no prediction — force prediction
            return 0, confidence, trace

    # Max steps reached — force prediction based on last reasoning
    return 0, confidence, trace


# ===================== MAIN LOSO LOOP =====================

async def main():
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Load data
    text_df = load_modality_texts()
    feat_df = pd.read_csv(RESULTS_DIR / 'all_subjects_features.csv')
    feat_df['category'] = feat_df['category'].astype(int)
    feat_df[TOP_FEATURES] = feat_df[TOP_FEATURES].fillna(0)

    # Align indices
    text_df.index = feat_df.index

    subjects = sorted(feat_df['P_ID'].unique())
    all_results = []
    all_sample_outputs = []

    print(f"{'='*60}")
    print(f"  ReAct Agent — Full Toolkit — LOSO-CV")
    print(f"  Tools: LR + sensor text + summary stats")
    print(f"  Max {MAX_STEPS} steps, confidence threshold {CONFIDENCE_THRESHOLD}")
    print(f"{'='*60}")

    t0 = time.time()
    total_api_calls = 0

    for fold_i, test_subj in enumerate(subjects):
        t_sub = time.time()

        train_mask = feat_df['P_ID'] != test_subj
        test_mask = feat_df['P_ID'] == test_subj

        # Subsample test
        test_feat = feat_df[test_mask]
        test_text = text_df[test_mask]
        rng = np.random.RandomState(SEED + fold_i)
        parts = []
        text_parts = []
        for cat in [0, 1]:
            pool_idx = test_feat[test_feat['category'] == cat].index
            n = min(SUBSAMPLE_PER_CLASS, len(pool_idx))
            if n > 0:
                chosen = rng.choice(pool_idx, size=n, replace=False)
                parts.append(test_feat.loc[chosen])
                text_parts.append(test_text.loc[chosen])
        test_feat_sub = pd.concat(parts)
        test_text_sub = pd.concat(text_parts)

        # Build toolkit for this fold
        toolkit = ToolKit(feat_df[train_mask], feat_df[train_mask]['category'].values)

        # Run agent on each sample
        tasks = []
        indices = []
        for idx in test_feat_sub.index:
            tasks.append(run_react_agent(client, semaphore, toolkit, feat_df.loc[idx], text_df.loc[idx]))
            indices.append(idx)

        results = await asyncio.gather(*tasks)

        y_true = test_feat_sub['category'].values
        y_pred = []
        for idx, (pred, conf, trace) in zip(indices, results):
            y_pred.append(pred)
            total_api_calls += len(trace)

            all_sample_outputs.append({
                'P_ID': test_subj,
                'true_label': int(feat_df.loc[idx, 'category']),
                'prediction': pred,
                'confidence': conf,
                'n_steps': len(trace),
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
        print(f"  [{fold_i+1:2d}/38] {test_subj}: "
              f"BalAcc={metrics['balanced_accuracy']:.3f}, "
              f"AvgSteps={metrics['avg_steps']:.1f} "
              f"({metrics['n_test']} samples, {elapsed_sub:.1f}s)", flush=True)

        # Incremental save
        pd.DataFrame(all_results).to_csv(RESULTS_DIR / 'results_react_full_toolkit.csv', index=False)
        pd.DataFrame(all_sample_outputs).to_csv(RESULTS_DIR / 'react_full_toolkit_per_sample.csv', index=False)

    elapsed = time.time() - t0
    print(f"\n  Total: {total_api_calls} API calls in {elapsed:.1f}s")

    # Final save
    rdf = pd.DataFrame(all_results)
    rdf.to_csv(RESULTS_DIR / 'results_react_full_toolkit.csv', index=False)

    preds_df = pd.DataFrame(all_sample_outputs)
    preds_df.to_csv(RESULTS_DIR / 'react_full_toolkit_per_sample.csv', index=False)

    # Save example traces
    examples = []
    for row in all_sample_outputs[:20]:
        examples.append({
            'P_ID': row['P_ID'],
            'true_label': row['true_label'],
            'prediction': row['prediction'],
            'confidence': row['confidence'],
            'n_steps': row['n_steps'],
            'trace': json.loads(row['trace']),
        })
    with open(RESULTS_DIR / 'react_example_traces.json', 'w') as f:
        json.dump(examples, f, indent=2)

    macro = {m: rdf[m].mean() for m in ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']}
    print(f"\n  MACRO AVERAGES:")
    for m, v in macro.items():
        print(f"    {m:20s}: {v:.4f}")
    print(f"    {'Avg steps/sample':20s}: {rdf['avg_steps'].mean():.2f}")
    print(f"    {'Total API calls':20s}: {total_api_calls}")
    print(f"    {'ML Baseline (LR)':20s}: 0.5653")


if __name__ == '__main__':
    asyncio.run(main())
