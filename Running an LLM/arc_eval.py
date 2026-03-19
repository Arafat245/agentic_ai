#!/usr/bin/env python3
"""
ARC (AI2 Reasoning Challenge) Evaluation Script

Evaluates LLMs on the ARC benchmark - a multiple-choice science reasoning dataset.
Supports ARC-Easy and ARC-Challenge configurations.

Output format is compatible with create_graphs.py for cross-dataset comparison.
Dataset: https://huggingface.co/datasets/allenai/ai2_arc

Usage:
  python3 arc_eval.py --model meta-llama/Llama-3.2-1B-Instruct --device cuda --config ARC-Easy
  python3 arc_eval.py --model Qwen/Qwen2.5-0.5B-Instruct --device cuda --config ARC-Challenge --max_examples 500
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


CHOICES = ["A", "B", "C", "D"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LLM on ARC benchmark")
    p.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Hugging Face model id",
    )
    p.add_argument(
        "--device",
        choices=["cuda", "cpu", "mps", "auto"],
        default="auto",
        help="Execution device",
    )
    p.add_argument(
        "--quant",
        choices=["none", "4bit", "8bit"],
        default="none",
        help="Quantization mode (CUDA only)",
    )
    p.add_argument(
        "--config",
        choices=["ARC-Easy", "ARC-Challenge"],
        default="ARC-Easy",
        help="ARC dataset configuration",
    )
    p.add_argument(
        "--split",
        choices=["test", "validation"],
        default="test",
        help="Dataset split to evaluate on",
    )
    p.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Cap number of examples for quick testing",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save results",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    return p.parse_args()


def resolve_device(preferred: str) -> str:
    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preferred == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def quant_bits_from_args(q: str) -> Optional[int]:
    if q == "none":
        return None
    if q == "4bit":
        return 4
    if q == "8bit":
        return 8
    raise ValueError(f"Unknown quant mode: {q}")


def get_quant_config(quant_bits: Optional[int]) -> Optional[BitsAndBytesConfig]:
    if quant_bits is None:
        return None
    if quant_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if quant_bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    raise ValueError(f"Invalid quant_bits: {quant_bits}")


def format_arc_prompt(question: str, choices_labels: List[str], choices_text: List[str]) -> str:
    """Format ARC question as multiple choice (same format as MMLU for consistency)."""
    prompt = f"{question}\n\n"
    for label, text in zip(choices_labels, choices_text):
        prompt += f"{label}. {text}\n"
    prompt += "\nAnswer:"
    return prompt


def _choice_token_ids(tokenizer: AutoTokenizer) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    for c in CHOICES:
        ids = []
        for variant in [c, f" {c}"]:
            tok = tokenizer.encode(variant, add_special_tokens=False)
            if len(tok) == 1:
                ids.append(tok[0])
        if not ids:
            ids = [tokenizer.encode(c, add_special_tokens=False)[0]]
        mapping[c] = list(dict.fromkeys(ids))
    return mapping


def predict_choice(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    choice_ids: Dict[str, List[int]],
) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model(**inputs)
        next_logits = out.logits[0, -1]

    best_choice = "A"
    best_score = None
    for c in CHOICES:
        ids = choice_ids[c]
        score = max(float(next_logits[i]) for i in ids)
        if best_score is None or score > best_score:
            best_score = score
            best_choice = c
    return best_choice


def main() -> str:
    args = parse_args()
    torch.manual_seed(args.seed)

    quant_bits = quant_bits_from_args(args.quant)
    device = resolve_device(args.device)

    if quant_bits is not None and device != "cuda":
        print("❌ Quantization requires CUDA. Use --device cuda or --quant none.")
        sys.exit(1)

    print("=" * 70)
    print("ARC Evaluation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print("=" * 70)

    # Load tokenizer and model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = get_quant_config(quant_bits)
    if quant_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quant_config,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
    else:
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=torch.float16,
                low_cpu_mem_usage=False,
            ).to("cuda")
        elif device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=torch.float16,
                low_cpu_mem_usage=False,
            ).to("mps")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=torch.float32,
                low_cpu_mem_usage=False,
            ).to("cpu")

    model.eval()
    print("✓ Model loaded")

    # Load ARC dataset
    print(f"\nLoading {args.config} ({args.split})...")
    dataset = load_dataset("allenai/ai2_arc", args.config, split=args.split)

    if args.max_examples is not None:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
        print(f"  Using {len(dataset)} examples (capped)")

    choice_ids = _choice_token_ids(tokenizer)

    correct = 0
    total = 0

    if device == "cuda":
        torch.cuda.synchronize()
    eval_start_wall = time.perf_counter()
    eval_start_cpu = time.process_time()

    for ex in tqdm(dataset, desc="Evaluating"):
        question = ex["question"]
        choices = ex["choices"]
        # ARC format: choices = {"label": ["A","B",...], "text": ["...","...",...]}
        labels = choices["label"] if isinstance(choices, dict) else [c["label"] for c in choices]
        texts = choices["text"] if isinstance(choices, dict) else [c["text"] for c in choices]
        correct_answer = ex["answerKey"]

        # Normalize answer key (sometimes comes as "A" or "1")
        if correct_answer not in CHOICES:
            ans_map = {"1": "A", "2": "B", "3": "C", "4": "D"}
            correct_answer = ans_map.get(str(correct_answer), "A")

        prompt = format_arc_prompt(question, labels, texts)
        pred = predict_choice(model, tokenizer, prompt, choice_ids)

        if pred == correct_answer:
            correct += 1
        total += 1

    if device == "cuda":
        torch.cuda.synchronize()
    eval_wall_s = time.perf_counter() - eval_start_wall
    eval_cpu_s = time.process_time() - eval_start_cpu

    accuracy = (correct / total * 100.0) if total else 0.0

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.config}")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Eval wall time: {eval_wall_s:.2f} s")
    print(f"Eval CPU time: {eval_cpu_s:.2f} s")
    print("=" * 70)

    # Save results (format compatible with create_graphs cross-dataset)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace("/", "__")
    out_name = f"arc_results_{safe_model}_{args.config.replace('-', '_')}_{device}_{ts}.json"
    out_path = os.path.join(args.output_dir, out_name)

    payload = {
        "model": args.model,
        "dataset": args.config,
        "benchmark": "ARC",
        "device": device,
        "timestamp": ts,
        "overall_accuracy": accuracy,
        "total_correct": correct,
        "total_questions": total,
        "subject_results": [{"subject": args.config, "correct": correct, "total": total, "accuracy": accuracy}],
        "timing": {"eval_wall_s": eval_wall_s, "eval_cpu_s": eval_cpu_s},
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n✓ Results saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
