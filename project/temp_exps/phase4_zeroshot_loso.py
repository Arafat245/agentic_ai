"""
Phase 4: Zero-Shot & Few-Shot LOSO Classification with Open-Source LLMs
- Loads text templates (feature values embedded in natural language)
- Classifies using generation + sequence-level log-probability scoring
- Evaluates with Leave-One-Subject-Out Cross-Validation
- Tests multiple models and shot counts
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
from pathlib import Path
import time
import warnings
import gc

warnings.filterwarnings('ignore')

PROJECT_DIR = Path('/mnt/sdb/arafat/agentic_ai/project')
TEXT_FEATURES_FILE = PROJECT_DIR / 'temp_exps' / 'text_features.csv'
RESULTS_DIR = PROJECT_DIR / 'temp_exps'

GPU_ID = 0
SEED = 42
np.random.seed(SEED)

SYSTEM_PROMPT = (
    "You are analyzing smartwatch sensor data to detect social interactions. "
    "You will receive sensor measurements from a 16-second window recorded "
    "on a participant's wrist. Based on the movement patterns, heart rate "
    "variability, and environmental light conditions, determine whether the "
    "participant was engaged in a social interaction.\n\n"
    "Social interactions include: conversations, group activities, meetings, "
    "eating with others, or any face-to-face engagement with other people.\n\n"
    "Respond with ONLY: \"interaction\" or \"no_interaction\"."
)

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
]

SHOT_COUNTS = [0, 5]


def load_data():
    df = pd.read_csv(TEXT_FEATURES_FILE)
    df['category'] = df['category'].astype(int)
    return df


def get_few_shot_examples(train_df, k, rng):
    """Select k balanced examples per class from training data."""
    if k == 0:
        return []
    examples = []
    for cat in [0, 1]:
        pool = train_df[train_df['category'] == cat]
        selected = pool.sample(n=min(k, len(pool)), random_state=rng)
        for _, row in selected.iterrows():
            label = "interaction" if row['category'] == 1 else "no_interaction"
            examples.append((row['text_description'], label))
    rng_obj = np.random.RandomState(rng)
    rng_obj.shuffle(examples)
    return examples


def build_messages(text, few_shot_examples=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot_examples:
        for ex_text, ex_label in few_shot_examples:
            messages.append({"role": "user", "content": ex_text})
            messages.append({"role": "assistant", "content": ex_label})
    messages.append({"role": "user", "content": text})
    return messages


def score_sequence(model, tokenizer, messages, answer_text, device):
    """Compute the log-probability of generating a specific answer given the prompt."""
    # Build full sequence: prompt + answer
    full_messages = messages + [{"role": "assistant", "content": answer_text}]
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

    # Build prompt-only to find where answer starts
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
    prompt_len = prompt_ids['input_ids'].shape[1]

    with torch.inference_mode():
        outputs = model(**full_ids)
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    # Compute log-prob of answer tokens only
    # For token at position i, logits[i-1] predicts it
    input_ids = full_ids['input_ids'][0]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    answer_log_prob = 0.0
    n_answer_tokens = 0
    for i in range(prompt_len, len(input_ids)):
        token_id = input_ids[i].item()
        answer_log_prob += log_probs[i - 1, token_id].item()
        n_answer_tokens += 1

    # Normalize by number of answer tokens
    if n_answer_tokens > 0:
        answer_log_prob /= n_answer_tokens

    return answer_log_prob


def classify_sample(model, tokenizer, messages, device):
    """Classify by comparing sequence-level log-probs of both answers."""
    score_int = score_sequence(model, tokenizer, messages, "interaction", device)
    score_no = score_sequence(model, tokenizer, messages, "no_interaction", device)

    # Convert to probability via softmax over the two scores
    scores = torch.tensor([score_int, score_no])
    probs = torch.softmax(scores, dim=0)
    prob_interaction = probs[0].item()

    pred = 1 if prob_interaction > 0.5 else 0
    return pred, prob_interaction


def evaluate_subject(y_true, y_pred, y_prob):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['auc'] = np.nan
    return metrics


def run_loso(df, model, tokenizer, device, shot_count, model_name):
    subjects = sorted(df['P_ID'].unique())
    all_results = []

    print(f"\n{'='*60}")
    print(f"  {model_name} — {shot_count}-shot LOSO-CV ({len(subjects)} subjects)")
    print(f"{'='*60}")

    t0 = time.time()

    for i, test_subject in enumerate(subjects):
        t_sub = time.time()
        train_df = df[df['P_ID'] != test_subject]
        test_df = df[df['P_ID'] == test_subject]

        few_shot_examples = get_few_shot_examples(train_df, shot_count, rng=SEED + i)

        y_true = test_df['category'].values
        y_pred = []
        y_prob = []

        for _, row in test_df.iterrows():
            messages = build_messages(row['text_description'], few_shot_examples)
            pred, prob = classify_sample(model, tokenizer, messages, device)
            y_pred.append(pred)
            y_prob.append(prob)

        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        metrics = evaluate_subject(y_true, y_pred, y_prob)
        metrics['subject'] = test_subject
        metrics['n_test'] = len(y_true)
        metrics['n_pos'] = int(y_true.sum())
        metrics['n_neg'] = int(len(y_true) - y_true.sum())
        all_results.append(metrics)

        elapsed_sub = time.time() - t_sub
        print(f"  [{i+1:2d}/{len(subjects)}] {test_subject}: "
              f"Acc={metrics['accuracy']:.3f}, BalAcc={metrics['balanced_accuracy']:.3f}, "
              f"F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f} "
              f"({metrics['n_test']} samples, {elapsed_sub:.1f}s)")

    elapsed = time.time() - t0
    results_df = pd.DataFrame(all_results)

    metric_cols = ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 'auc']
    macro = {m: results_df[m].mean() for m in metric_cols}

    print(f"\n  MACRO AVERAGES:")
    for m, v in macro.items():
        print(f"    {m:20s}: {v:.4f}")
    print(f"    {'Time':20s}: {elapsed:.1f}s")

    return results_df, macro


def main():
    df = load_data()
    print(f"Loaded {len(df)} samples, {df['P_ID'].nunique()} subjects")
    print(f"Class distribution: {dict(df['category'].value_counts().sort_index())}")

    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_summaries = []

    for model_name in MODELS:
        print(f"\n{'#'*60}")
        print(f"  Loading model: {model_name}")
        print(f"{'#'*60}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map={"": device},
                trust_remote_code=True,
            )
            model.eval()

            for shot_count in SHOT_COUNTS:
                results_df, macro = run_loso(
                    df, model, tokenizer, device, shot_count, model_name
                )

                safe_name = model_name.split('/')[-1].lower().replace('-', '_')
                results_df.to_csv(
                    RESULTS_DIR / f'results_zeroshot_{safe_name}_{shot_count}shot.csv',
                    index=False
                )

                all_summaries.append({
                    'model': model_name.split('/')[-1],
                    'shots': shot_count,
                    **{k: f"{v:.4f}" for k, v in macro.items()},
                })

        except Exception as e:
            print(f"  ERROR loading {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            if 'model' in dir():
                del model
            gc.collect()
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(RESULTS_DIR / 'results_zeroshot_summary.csv', index=False)
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR / 'results_zeroshot_summary.csv'}")


if __name__ == '__main__':
    main()
