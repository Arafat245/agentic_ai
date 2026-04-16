"""
Phase 6 — Local Single-Agent Baseline (Qwen2.5-7B-Instruct on GPU 0)
1-shot LOSO-CV using log-probability scoring.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import time
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
from phase6_utils import (
    load_modality_texts, get_loso_splits, get_few_shot_examples,
    compute_metrics, save_subject_results, RESULTS_DIR, SEED,
    SYSTEM_PROMPT_SINGLE
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
GPU_ID = 1  # CUDA device 1 = A5000 (24GB, free)
MAX_SEQ_LEN = 2048


def build_chat_messages(text, few_shot_examples=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT_SINGLE}]
    if few_shot_examples:
        for ex_text, ex_label in few_shot_examples:
            messages.append({"role": "user", "content": ex_text})
            messages.append({"role": "assistant", "content": f"Prediction: {ex_label}\nReason: Based on the sensor patterns."})
    messages.append({"role": "user", "content": text})
    return messages


def score_sequence(model, tokenizer, messages, answer_text, device):
    """Compute normalized log-probability of generating answer_text given prompt."""
    full_messages = messages + [{"role": "assistant", "content": answer_text}]
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    prompt_len = prompt_ids['input_ids'].shape[1]

    with torch.inference_mode():
        outputs = model(**full_ids)
        logits = outputs.logits[0]

    input_ids = full_ids['input_ids'][0]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    answer_log_prob = 0.0
    n_tokens = 0
    for i in range(prompt_len, len(input_ids)):
        token_id = input_ids[i].item()
        answer_log_prob += log_probs[i - 1, token_id].item()
        n_tokens += 1

    if n_tokens > 0:
        answer_log_prob /= n_tokens

    return answer_log_prob


def classify_sample(model, tokenizer, messages, device):
    """Classify by comparing log-probs of 'interaction' vs 'no_interaction'."""
    score_int = score_sequence(model, tokenizer, messages, "Prediction: interaction", device)
    score_no = score_sequence(model, tokenizer, messages, "Prediction: no_interaction", device)

    scores = torch.tensor([score_int, score_no])
    probs = torch.softmax(scores, dim=0)
    prob_interaction = probs[0].item()

    pred = 1 if prob_interaction > 0.5 else 0
    return pred, prob_interaction


def main():
    df = load_modality_texts()
    print(f"Loaded {len(df)} samples, {df['P_ID'].nunique()} subjects")

    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")

    shot_count = 1
    all_results = []

    print(f"\n{'='*60}")
    print(f"  {MODEL_NAME.split('/')[-1]} — {shot_count}-shot LOSO-CV")
    print(f"{'='*60}")

    t0 = time.time()

    for fold_i, (test_subj, train_df, test_df) in enumerate(get_loso_splits(df)):
        t_sub = time.time()

        few_shot = get_few_shot_examples(train_df, shot_count, modality='all', seed=SEED + fold_i)

        y_true = test_df['category'].values
        y_pred = []
        y_prob = []

        for _, row in test_df.iterrows():
            messages = build_chat_messages(row['text_all'], few_shot)
            pred, prob = classify_sample(model, tokenizer, messages, device)
            y_pred.append(pred)
            y_prob.append(prob)

        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics['subject'] = test_subj
        metrics['n_test'] = len(y_true)
        metrics['n_pos'] = int(y_true.sum())
        all_results.append(metrics)

        elapsed_sub = time.time() - t_sub
        print(f"  [{fold_i+1:2d}/38] {test_subj}: "
              f"BalAcc={metrics['balanced_accuracy']:.3f}, F1={metrics['f1']:.3f}, "
              f"AUC={metrics['auc']:.3f} ({metrics['n_test']} samples, {elapsed_sub:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/3600:.1f}h)")

    output_path = RESULTS_DIR / 'results_local_qwen7b_1shot.csv'
    macro = save_subject_results(all_results, output_path)

    print(f"\n  Qwen-7B 1-shot Balanced Accuracy: {macro['balanced_accuracy']:.4f}")
    print(f"  ML Baseline (LR):                  0.5761")


if __name__ == '__main__':
    main()
