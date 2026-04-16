"""
Phase 7 — Local Multi-Agent with any HuggingFace model (subsampled LOSO-CV).
Full ConSensus pipeline: modality agents + statistical + semantic + hybrid fusion.
Usage: python phase7_local_multi_agent.py <model_name>
Example: python phase7_local_multi_agent.py meta-llama/Llama-3.1-8B-Instruct
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import time
import sys
import argparse
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
from phase6_utils import (
    load_modality_texts, get_loso_splits, get_few_shot_examples,
    subsample_test, parse_prediction, compute_metrics, save_subject_results,
    statistical_fusion, RESULTS_DIR, SEED,
    MODALITY_SYSTEM_PROMPTS, SYSTEM_PROMPT_SEMANTIC_FUSION, SYSTEM_PROMPT_HYBRID_FUSION,
)

GPU_ID = 1  # CUDA device 1 = A5000 (24GB)
MAX_NEW_TOKENS = 80
SUBSAMPLE_PER_CLASS = 25


def generate_response(model, tokenizer, messages, device):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536).to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def run_modality_agent(model, tokenizer, device, modality, text, few_shot):
    sys_prompt = MODALITY_SYSTEM_PROMPTS[modality]
    messages = [{"role": "system", "content": sys_prompt}]
    if few_shot:
        for ex_text, ex_label in few_shot:
            messages.append({"role": "user", "content": ex_text})
            messages.append({"role": "assistant", "content": f"Prediction: {ex_label}\nReason: Based on the {modality} patterns."})
    messages.append({"role": "user", "content": text})
    response = generate_response(model, tokenizer, messages, device)
    return parse_prediction(response)


def run_semantic_fusion(model, tokenizer, device, acc_pred, acc_reason, ppg_pred, ppg_reason, light_pred, light_reason):
    acc_l = "interaction" if acc_pred == 1 else "no_interaction"
    ppg_l = "interaction" if ppg_pred == 1 else "no_interaction"
    light_l = "interaction" if light_pred == 1 else "no_interaction"
    user_content = (
        f"Accelerometer Agent:\n  Prediction: {acc_l}\n  Reason: {acc_reason}\n\n"
        f"PPG Agent:\n  Prediction: {ppg_l}\n  Reason: {ppg_reason}\n\n"
        f"Light Agent:\n  Prediction: {light_l}\n  Reason: {light_reason}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_SEMANTIC_FUSION},
        {"role": "user", "content": user_content},
    ]
    response = generate_response(model, tokenizer, messages, device)
    return parse_prediction(response)


def run_hybrid_fusion(model, tokenizer, device, sem_pred, sem_reason, stat_pred, acc_pred, ppg_pred, light_pred):
    sem_l = "interaction" if sem_pred == 1 else "no_interaction"
    stat_l = "interaction" if stat_pred == 1 else "no_interaction"
    acc_l = "interaction" if acc_pred == 1 else "no_interaction"
    ppg_l = "interaction" if ppg_pred == 1 else "no_interaction"
    light_l = "interaction" if light_pred == 1 else "no_interaction"
    user_content = (
        f"Semantic Fusion Agent:\n  Prediction: {sem_l}\n  Reason: {sem_reason}\n\n"
        f"Statistical Fusion (majority vote):\n  Prediction: {stat_l}\n"
        f"  Votes: ACC={acc_l}, PPG={ppg_l}, Light={light_l}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_HYBRID_FUSION},
        {"role": "user", "content": user_content},
    ]
    response = generate_response(model, tokenizer, messages, device)
    return parse_prediction(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='HuggingFace model name')
    args = parser.parse_args()

    model_name = args.model_name
    short_name = model_name.split('/')[-1].lower().replace('-', '_').replace('.', '_')

    df = load_modality_texts()
    print(f"Loaded {len(df)} samples, {df['P_ID'].nunique()} subjects")

    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map={"": device}, trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")

    modalities = ['acc', 'ppg', 'light']
    method_names = ['acc', 'ppg', 'light', 'stat_fusion', 'semantic_fusion', 'hybrid_fusion']
    results = {m: [] for m in method_names}
    all_sample_outputs = []

    print(f"\n{'='*60}")
    print(f"  Local Multi-Agent — {model_name.split('/')[-1]} — Subsampled LOSO-CV")
    print(f"  ({SUBSAMPLE_PER_CLASS} per class per subject)")
    print(f"{'='*60}")

    t0 = time.time()
    total_samples = 0

    for fold_i, (test_subj, train_df, test_df) in enumerate(get_loso_splits(df)):
        t_sub = time.time()
        test_sub = subsample_test(test_df, n_per_class=SUBSAMPLE_PER_CLASS, seed=SEED + fold_i)
        y_true = test_sub['category'].values
        total_samples += len(test_sub)

        few_shots = {}
        for mod in modalities:
            few_shots[mod] = get_few_shot_examples(train_df, 1, modality=mod, seed=SEED + fold_i)

        all_preds = {m: [] for m in method_names}

        for _, row in test_sub.iterrows():
            mod_preds = {}
            mod_reasons = {}
            for mod in modalities:
                pred, reason = run_modality_agent(model, tokenizer, device, mod, row[f'text_{mod}'], few_shots[mod])
                mod_preds[mod] = pred
                mod_reasons[mod] = reason
                all_preds[mod].append(pred)

            stat_pred = statistical_fusion(mod_preds['acc'], mod_preds['ppg'], mod_preds['light'])
            all_preds['stat_fusion'].append(stat_pred)

            sem_pred, sem_reason = run_semantic_fusion(
                model, tokenizer, device,
                mod_preds['acc'], mod_reasons['acc'],
                mod_preds['ppg'], mod_reasons['ppg'],
                mod_preds['light'], mod_reasons['light'],
            )
            all_preds['semantic_fusion'].append(sem_pred)

            hyb_pred, hyb_reason = run_hybrid_fusion(
                model, tokenizer, device,
                sem_pred, sem_reason, stat_pred,
                mod_preds['acc'], mod_preds['ppg'], mod_preds['light'],
            )
            all_preds['hybrid_fusion'].append(hyb_pred)

            all_sample_outputs.append({
                'P_ID': row['P_ID'], 'true_label': row['category'],
                'acc_pred': mod_preds['acc'], 'acc_reason': mod_reasons['acc'],
                'ppg_pred': mod_preds['ppg'], 'ppg_reason': mod_reasons['ppg'],
                'light_pred': mod_preds['light'], 'light_reason': mod_reasons['light'],
                'stat_fusion': stat_pred, 'semantic_fusion': sem_pred,
                'semantic_reason': sem_reason, 'hybrid_fusion': hyb_pred,
            })

        for method in method_names:
            y_pred = np.array(all_preds[method], dtype=int).flatten()
            if len(y_pred) != len(y_true) or len(y_pred) == 0:
                print(f"    WARNING: {method} pred shape mismatch, skipping")
                continue
            metrics = compute_metrics(y_true, y_pred)
            metrics['subject'] = test_subj
            metrics['n_test'] = len(y_true)
            metrics['n_pos'] = int(y_true.sum())
            results[method].append(metrics)

        elapsed_sub = time.time() - t_sub
        ba_hyb = results['hybrid_fusion'][-1]['balanced_accuracy']
        ba_stat = results['stat_fusion'][-1]['balanced_accuracy']
        print(f"  [{fold_i+1:2d}/38] {test_subj}: "
              f"Hybrid={ba_hyb:.3f}, Stat={ba_stat:.3f} "
              f"({len(test_sub)} samples, {elapsed_sub:.1f}s)", flush=True)

        # === INCREMENTAL SAVE after each subject ===
        sample_df = pd.DataFrame(all_sample_outputs)
        sample_df.to_csv(RESULTS_DIR / f'{short_name}_multi_agent_per_sample.csv', index=False)
        for method, res_list in results.items():
            if res_list:
                pd.DataFrame(res_list).to_csv(
                    RESULTS_DIR / f'results_{short_name}_multi_{method}.csv', index=False)

    elapsed = time.time() - t0
    print(f"\n  Total: {total_samples} samples in {elapsed:.1f}s ({elapsed/3600:.1f}h)")

    # Final save
    sample_df = pd.DataFrame(all_sample_outputs)
    sample_df.to_csv(RESULTS_DIR / f'{short_name}_multi_agent_per_sample.csv', index=False)
    print(f"  Saved {len(sample_df)} per-sample predictions")

    macros = {}
    for method, res_list in results.items():
        output_path = RESULTS_DIR / f'results_{short_name}_multi_{method}.csv'
        print(f"\n  --- {method.upper()} ---")
        macro = save_subject_results(res_list, output_path)
        macros[method] = macro

    print(f"\n{'='*60}")
    print(f"  {model_name.split('/')[-1]} MULTI-AGENT SUMMARY (Balanced Accuracy)")
    print(f"{'='*60}")
    for method, macro in macros.items():
        print(f"  {method:20s}: {macro['balanced_accuracy']:.4f}")
    print(f"  {'ML Baseline':20s}: 0.5761")


if __name__ == '__main__':
    main()
