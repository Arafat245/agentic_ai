"""
Phase 5: LoRA Fine-Tuning with LOSO-CV for Social Interaction Detection
- Uses text templates as input to small open-source LLMs
- Fine-tunes with LoRA in Leave-One-Subject-Out manner
- Evaluates classification performance per subject
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
from pathlib import Path
import time
import gc
import warnings

warnings.filterwarnings('ignore')

PROJECT_DIR = Path('/mnt/sdb/arafat/agentic_ai/project')
TEXT_FEATURES_FILE = PROJECT_DIR / 'temp_exps' / 'text_features.csv'
RESULTS_DIR = PROJECT_DIR / 'temp_exps'
CHECKPOINT_DIR = PROJECT_DIR / 'temp_exps' / 'loso_ft'

GPU_ID = 0
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

SYSTEM_PROMPT = (
    "You are analyzing smartwatch sensor data to detect social interactions. "
    "You will receive sensor measurements from a 16-second window recorded "
    "on a participant's wrist. Based on the movement patterns, heart rate "
    "variability, and environmental light conditions, determine whether the "
    "participant was engaged in a social interaction.\n\n"
    "Social interactions include: conversations, group activities, meetings, "
    "eating with others, or any face-to-face engagement with other people.\n\n"
    "Respond with ONLY one word: \"interaction\" or \"no_interaction\"."
)

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

TRAIN_EPOCHS = 3
BATCH_SIZE = 16
GRAD_ACCUM = 2
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 512


# ===================== DATASET =====================

class SensorTextDataset(Dataset):
    """Dataset that tokenizes chat messages and masks loss to assistant tokens only."""

    def __init__(self, texts, labels, tokenizer, max_length=MAX_SEQ_LEN):
        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        for text, label in zip(texts, labels):
            answer = "interaction" if label == 1 else "no_interaction"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
                {"role": "assistant", "content": answer},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            encoding = tokenizer(
                full_text, truncation=True, max_length=max_length,
                padding=False, return_tensors="pt"
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attn_mask = encoding['attention_mask'].squeeze(0)

            # Create labels: mask everything except the answer tokens
            # Find where the assistant answer starts
            # Build prompt without the answer to find the split point
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(
                prompt_text, truncation=True, max_length=max_length,
                padding=False, return_tensors="pt"
            )
            prompt_len = prompt_ids['input_ids'].shape[1]

            # Labels: -100 for prompt tokens, actual ids for answer tokens
            label_ids = input_ids.clone()
            label_ids[:prompt_len] = -100

            self.input_ids.append(input_ids)
            self.attention_masks.append(attn_mask)
            self.labels.append(label_ids)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx],
        }


class PaddingCollator:
    """Pad batch to max length with proper padding for input_ids, attention_mask, and labels."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, batch):
        max_len = max(item['input_ids'].shape[0] for item in batch)
        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            seq_len = item['input_ids'].shape[0]
            pad_len = max_len - seq_len
            input_ids.append(torch.cat([item['input_ids'], torch.full((pad_len,), self.pad_id)]))
            attention_masks.append(torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)]))
            labels.append(torch.cat([item['labels'], torch.full((pad_len,), -100)]))

        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'labels': torch.stack(labels),
        }


# ===================== HELPERS =====================

def load_data():
    df = pd.read_csv(TEXT_FEATURES_FILE)
    df['category'] = df['category'].astype(int)
    return df


def oversample_minority(texts, labels):
    """Oversample minority class to balance the dataset."""
    texts = list(texts)
    labels = list(labels)
    counts = {0: labels.count(0), 1: labels.count(1)}
    majority = max(counts, key=counts.get)
    minority = 1 - majority
    diff = counts[majority] - counts[minority]

    minority_indices = [i for i, l in enumerate(labels) if l == minority]
    rng = np.random.RandomState(SEED)
    extra_indices = rng.choice(minority_indices, size=diff, replace=True)

    for idx in extra_indices:
        texts.append(texts[idx])
        labels.append(labels[idx])

    return texts, labels


def score_sequence(model, tokenizer, messages, answer_text, device):
    """Compute the log-probability of generating a specific answer given the prompt."""
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
    n_answer_tokens = 0
    for i in range(prompt_len, len(input_ids)):
        token_id = input_ids[i].item()
        answer_log_prob += log_probs[i - 1, token_id].item()
        n_answer_tokens += 1

    if n_answer_tokens > 0:
        answer_log_prob /= n_answer_tokens

    return answer_log_prob


def classify_sample(model, tokenizer, text, device):
    """Classify by comparing sequence-level log-probs of both answers."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    score_int = score_sequence(model, tokenizer, messages, "interaction", device)
    score_no = score_sequence(model, tokenizer, messages, "no_interaction", device)

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


# ===================== MAIN LOSO LOOP =====================

def run_finetune_loso(df, model_name):
    """Run full LOSO fine-tuning for a given model."""
    subjects = sorted(df['P_ID'].unique())
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

    short_name = model_name.split('/')[-1]
    print(f"\n{'#'*60}")
    print(f"  LoRA Fine-Tuning LOSO: {short_name}")
    print(f"  {len(subjects)} folds, {TRAIN_EPOCHS} epochs each")
    print(f"{'#'*60}")

    all_results = []
    total_t0 = time.time()

    for fold_i, test_subject in enumerate(subjects):
        fold_t0 = time.time()
        print(f"\n--- Fold [{fold_i+1}/{len(subjects)}]: Test={test_subject} ---")

        train_df = df[df['P_ID'] != test_subject]
        test_df = df[df['P_ID'] == test_subject]

        train_texts = train_df['text_description'].tolist()
        train_labels = train_df['category'].tolist()

        # Oversample minority class
        train_texts, train_labels = oversample_minority(train_texts, train_labels)
        print(f"  Train: {len(train_texts)} samples (after oversampling), Test: {len(test_df)}")

        # Load fresh base model + LoRA for each fold
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": device},
            trust_remote_code=True,
        )
        model = get_peft_model(model, LORA_CONFIG)
        model.print_trainable_parameters()

        # Create dataset
        train_dataset = SensorTextDataset(train_texts, train_labels, tokenizer)
        collator = PaddingCollator(tokenizer)

        # Training
        output_dir = CHECKPOINT_DIR / short_name / test_subject
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=TRAIN_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            fp16=True,
            logging_steps=100,
            save_strategy="no",
            seed=SEED,
            dataloader_num_workers=4,
            report_to="none",
            max_grad_norm=1.0,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
        )
        trainer.train()

        # Evaluate on held-out subject
        model.eval()

        y_true = test_df['category'].values
        y_pred = []
        y_prob = []

        for _, row in test_df.iterrows():
            pred, prob = classify_sample(model, tokenizer, row['text_description'], device)
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

        fold_time = time.time() - fold_t0
        print(f"  Result: Acc={metrics['accuracy']:.3f}, BalAcc={metrics['balanced_accuracy']:.3f}, "
              f"F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f} ({fold_time:.1f}s)")

        # Clean up for next fold
        del model, trainer, train_dataset, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - total_t0
    results_df = pd.DataFrame(all_results)

    # Macro averages
    metric_cols = ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 'auc']
    macro = {m: results_df[m].mean() for m in metric_cols}

    print(f"\n{'='*60}")
    print(f"  {short_name} LoRA Fine-Tune — MACRO AVERAGES")
    print(f"{'='*60}")
    for m, v in macro.items():
        print(f"    {m:20s}: {v:.4f}")
    print(f"    {'Total time':20s}: {total_time:.1f}s ({total_time/3600:.1f}h)")

    # Save results
    safe_name = short_name.lower().replace('-', '_')
    results_df.to_csv(RESULTS_DIR / f'results_finetune_{safe_name}_lora.csv', index=False)

    return results_df, macro


def main():
    df = load_data()
    print(f"Loaded {len(df)} samples, {df['P_ID'].nunique()} subjects")
    print(f"Class distribution: {dict(df['category'].value_counts().sort_index())}")

    all_summaries = []

    for model_name in MODELS:
        try:
            results_df, macro = run_finetune_loso(df, model_name)
            all_summaries.append({
                'model': model_name.split('/')[-1],
                'method': 'LoRA fine-tune',
                **{k: f"{v:.4f}" for k, v in macro.items()},
            })
        except Exception as e:
            print(f"ERROR with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(RESULTS_DIR / 'results_finetune_summary.csv', index=False)
        print(f"\n{'='*60}")
        print("  FINAL FINE-TUNING SUMMARY")
        print(f"{'='*60}")
        print(summary_df.to_string(index=False))
        print(f"\nSaved to {RESULTS_DIR / 'results_finetune_summary.csv'}")


if __name__ == '__main__':
    main()
