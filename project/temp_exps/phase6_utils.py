"""
Shared utilities for Phase 6 ConSensus-style experiments.
"""

import pandas as pd
import numpy as np
import asyncio
import time
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root and temp_exps
load_dotenv(Path(__file__).parent.parent / '.env')
load_dotenv(Path(__file__).parent / '.env')

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)

PROJECT_DIR = Path('/mnt/sdb/arafat/agentic_ai/project')
MODALITY_TEXT_FILE = PROJECT_DIR / 'temp_exps' / 'modality_text_features.csv'
RESULTS_DIR = PROJECT_DIR / 'temp_exps'
SEED = 42


# ===================== DATA LOADING =====================

def load_modality_texts():
    """Load the per-modality text features CSV."""
    df = pd.read_csv(MODALITY_TEXT_FILE)
    df['category'] = df['category'].astype(int)
    return df


def get_loso_splits(df):
    """Yield (test_subject, train_df, test_df) for each LOSO fold."""
    subjects = sorted(df['P_ID'].unique())
    for subj in subjects:
        train_df = df[df['P_ID'] != subj]
        test_df = df[df['P_ID'] == subj]
        yield subj, train_df, test_df


def get_few_shot_examples(train_df, k, modality='all', seed=SEED):
    """
    Select k examples per class from training data for few-shot prompting.
    modality: 'acc', 'ppg', 'light', or 'all'
    Returns list of (text, label_str) tuples.
    """
    if k == 0:
        return []

    text_col = f'text_{modality}'
    examples = []
    rng = np.random.RandomState(seed)

    for cat in [0, 1]:
        pool = train_df[train_df['category'] == cat]
        selected = pool.sample(n=min(k, len(pool)), random_state=rng)
        for _, row in selected.iterrows():
            label = "interaction" if row['category'] == 1 else "no_interaction"
            examples.append((row[text_col], label))

    rng.shuffle(examples)
    return examples


def subsample_test(test_df, n_per_class=25, seed=SEED):
    """Subsample test set to n_per_class samples per class (stratified)."""
    rng = np.random.RandomState(seed)
    parts = []
    for cat in [0, 1]:
        pool = test_df[test_df['category'] == cat]
        n = min(n_per_class, len(pool))
        if n > 0:
            parts.append(pool.sample(n=n, random_state=rng))
    if parts:
        return pd.concat(parts).sort_index()
    return test_df


# ===================== METRICS =====================

def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute standard classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = np.nan
    else:
        metrics['auc'] = np.nan
    return metrics


def save_subject_results(results_list, output_path):
    """Save per-subject results to CSV and print macro averages."""
    df = pd.DataFrame(results_list)
    df.to_csv(output_path, index=False)

    metric_cols = ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 'auc']
    macro = {m: df[m].mean() for m in metric_cols if m in df.columns}

    print(f"\n  MACRO AVERAGES:")
    for m, v in macro.items():
        print(f"    {m:20s}: {v:.4f}")
    print(f"  Saved to {output_path}")

    return macro


# ===================== PREDICTION PARSING =====================

def parse_prediction(response_text):
    """
    Parse LLM response to extract prediction and reasoning.
    Returns (pred_int, reason_str).
    """
    if response_text is None:
        return 0, "no response"
    text = response_text.strip().lower()

    # Try structured format first: "Prediction: interaction"
    for line in response_text.strip().split('\n'):
        line_lower = line.strip().lower()
        if line_lower.startswith('prediction:'):
            val = line_lower.split(':', 1)[1].strip()
            if 'no_interaction' in val or 'no interaction' in val:
                reason = _extract_reason(response_text)
                return 0, reason
            elif 'interaction' in val:
                reason = _extract_reason(response_text)
                return 1, reason

    # Fallback: keyword search
    # Check no_interaction first (it contains "interaction")
    if 'no_interaction' in text or 'no interaction' in text:
        return 0, response_text.strip()
    elif 'interaction' in text:
        return 1, response_text.strip()

    # Default to majority class
    return 0, response_text.strip()


def _extract_reason(response_text):
    """Extract the Reason line from structured response."""
    for line in response_text.strip().split('\n'):
        if line.strip().lower().startswith('reason:'):
            return line.split(':', 1)[1].strip()
    return response_text.strip()


# ===================== PROMPT BUILDERS =====================

SYSTEM_PROMPT_SINGLE = (
    "You are analyzing smartwatch sensor data to detect social interactions. "
    "You will receive sensor measurements from a 16-second window recorded "
    "on a participant's wrist. Based on the movement patterns, heart rate "
    "variability, and environmental light conditions, determine whether the "
    "participant was engaged in a social interaction.\n\n"
    "Social interactions include: conversations, group activities, meetings, "
    "eating with others, or any face-to-face engagement with other people.\n\n"
    "Respond in this format:\n"
    "Prediction: [interaction or no_interaction]\n"
    "Reason: [one sentence]"
)

SYSTEM_PROMPT_ACC = (
    "You are an accelerometer specialist analyzing wrist movement patterns from "
    "a smartwatch. You will receive statistical features extracted from 16 seconds "
    "of tri-axial accelerometer data (X, Y, Z axes and magnitude). Based on the "
    "movement dynamics, frequency characteristics, and motion intensity, determine "
    "whether the wearer was likely engaged in a social interaction.\n\n"
    "Social interactions typically involve characteristic wrist movements from "
    "gesturing during conversation, eating with others, or group activities.\n\n"
    "Respond in this format:\n"
    "Prediction: [interaction or no_interaction]\n"
    "Reason: [one sentence explaining your reasoning based on accelerometer evidence]"
)

SYSTEM_PROMPT_PPG = (
    "You are a photoplethysmography (PPG) specialist analyzing heart rate and "
    "physiological signals from a smartwatch. You will receive PPG-derived features "
    "from a 16-second recording, including signal quality, heart rate variability "
    "(HRV) metrics, and pulse characteristics. Based on the physiological patterns, "
    "determine whether the wearer was likely engaged in a social interaction.\n\n"
    "Social interactions can affect autonomic nervous system activity, reflected in "
    "HRV changes and shifts in parasympathetic/sympathetic balance.\n\n"
    "Respond in this format:\n"
    "Prediction: [interaction or no_interaction]\n"
    "Reason: [one sentence explaining your reasoning based on PPG evidence]"
)

SYSTEM_PROMPT_LIGHT = (
    "You are an ambient light specialist analyzing environmental light conditions "
    "from a smartwatch light sensor. You will receive light features from a "
    "16-second recording, including brightness levels, variability, and change "
    "patterns. Based on the environmental light conditions, determine whether "
    "the wearer was likely in a social interaction setting.\n\n"
    "Social interactions often occur in specific lighting environments (indoor "
    "spaces, restaurants, meeting rooms) that differ from solitary activities.\n\n"
    "Respond in this format:\n"
    "Prediction: [interaction or no_interaction]\n"
    "Reason: [one sentence explaining your reasoning based on light evidence]"
)

SYSTEM_PROMPT_SEMANTIC_FUSION = (
    "You are a sensor fusion expert. Three modality-specific agents have "
    "independently analyzed a 16-second smartwatch recording to determine whether "
    "a social interaction occurred. Each agent only saw its own sensor data.\n\n"
    "Review their predictions and reasoning, then make a final determination by "
    "synthesizing the cross-modal evidence.\n\n"
    "Respond in this format:\n"
    "Prediction: [interaction or no_interaction]\n"
    "Reason: [one sentence synthesizing cross-modal evidence]"
)

SYSTEM_PROMPT_HYBRID_FUSION = (
    "You are making the final decision about whether a social interaction occurred "
    "during a 16-second smartwatch recording. You have two sources of information:\n"
    "1. Semantic Fusion: an agent that reasoned across all sensor modalities\n"
    "2. Statistical Fusion: majority vote of 3 modality-specific agents\n\n"
    "If both agree, follow them. If they disagree, weigh whether the semantic "
    "reasoning is well-grounded or whether the majority vote should take precedence.\n\n"
    "Respond in this format:\n"
    "Prediction: [interaction or no_interaction]"
)

MODALITY_SYSTEM_PROMPTS = {
    'acc': SYSTEM_PROMPT_ACC,
    'ppg': SYSTEM_PROMPT_PPG,
    'light': SYSTEM_PROMPT_LIGHT,
}


def build_messages_single(text, few_shot_examples=None):
    """Build messages for single-agent classification."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT_SINGLE}]
    if few_shot_examples:
        for ex_text, ex_label in few_shot_examples:
            messages.append({"role": "user", "content": ex_text})
            messages.append({"role": "assistant", "content": f"Prediction: {ex_label}\nReason: Based on the sensor patterns."})
    messages.append({"role": "user", "content": text})
    return messages


def build_messages_modality(modality, text, few_shot_examples=None):
    """Build messages for a modality-specific agent."""
    sys_prompt = MODALITY_SYSTEM_PROMPTS[modality]
    messages = [{"role": "system", "content": sys_prompt}]
    if few_shot_examples:
        for ex_text, ex_label in few_shot_examples:
            messages.append({"role": "user", "content": ex_text})
            messages.append({"role": "assistant", "content": f"Prediction: {ex_label}\nReason: Based on the {modality} patterns."})
    messages.append({"role": "user", "content": text})
    return messages


def build_messages_semantic_fusion(acc_pred, acc_reason, ppg_pred, ppg_reason, light_pred, light_reason):
    """Build messages for semantic fusion agent."""
    user_content = (
        f"Accelerometer Agent:\n"
        f"  Prediction: {acc_pred}\n"
        f"  Reason: {acc_reason}\n\n"
        f"PPG Agent:\n"
        f"  Prediction: {ppg_pred}\n"
        f"  Reason: {ppg_reason}\n\n"
        f"Light Agent:\n"
        f"  Prediction: {light_pred}\n"
        f"  Reason: {light_reason}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_SEMANTIC_FUSION},
        {"role": "user", "content": user_content},
    ]


def build_messages_hybrid_fusion(semantic_pred, semantic_reason, stat_pred, acc_pred, ppg_pred, light_pred):
    """Build messages for hybrid fusion agent."""
    user_content = (
        f"Semantic Fusion Agent:\n"
        f"  Prediction: {semantic_pred}\n"
        f"  Reason: {semantic_reason}\n\n"
        f"Statistical Fusion (majority vote):\n"
        f"  Prediction: {stat_pred}\n"
        f"  Votes: ACC={acc_pred}, PPG={ppg_pred}, Light={light_pred}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_HYBRID_FUSION},
        {"role": "user", "content": user_content},
    ]


# ===================== ASYNC API HELPERS =====================

async def call_anthropic(client, messages, semaphore, model="claude-haiku-4-5-20251001", max_retries=3):
    """Async Anthropic API call with retry and rate limiting."""
    system_msg = None
    user_messages = []
    for m in messages:
        if m['role'] == 'system':
            system_msg = m['content']
        else:
            user_messages.append(m)

    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.messages.create(
                    model=model,
                    max_tokens=100,
                    system=system_msg or "",
                    messages=user_messages,
                )
                return response.content[0].text
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                await asyncio.sleep(wait)
            else:
                return f"ERROR: {e}"


async def call_openai(client, messages, semaphore, model="gpt-4o-mini", max_retries=5):
    """Async OpenAI API call with retry and rate limiting."""
    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=80,
                    temperature=0,
                )
                return response.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            if '429' in err_str or 'rate_limit' in err_str.lower():
                wait = 2 ** attempt + 1
                await asyncio.sleep(wait)
            elif attempt < max_retries - 1:
                wait = 2 ** attempt
                await asyncio.sleep(wait)
            else:
                return f"ERROR: {e}"


def get_api_client():
    """
    Auto-detect which API key is available and return (call_fn, client, provider_name).
    Prioritizes OpenAI (cheaper, Anthropic credits may be exhausted).
    """
    anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
    openai_key = os.environ.get('OPENAI_API_KEY')

    if anthropic_key:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=anthropic_key)
        print("Using Anthropic API (Claude Haiku)")
        return call_anthropic, client, "claude-haiku"
    elif openai_key:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=openai_key)
        print("Using OpenAI API (GPT-4o-mini)")
        return call_openai, client, "gpt-4o-mini"
    else:
        raise ValueError(
            "No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.\n"
            "  export ANTHROPIC_API_KEY='your-key'\n"
            "  export OPENAI_API_KEY='your-key'"
        )


# ===================== STATISTICAL FUSION =====================

def statistical_fusion(acc_pred, ppg_pred, light_pred):
    """Majority vote of 3 modality predictions. Returns int (0 or 1)."""
    votes = [acc_pred, ppg_pred, light_pred]
    return 1 if sum(v == 1 for v in votes) >= 2 else 0
