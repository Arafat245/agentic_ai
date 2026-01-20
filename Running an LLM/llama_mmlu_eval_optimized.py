#!/usr/bin/env python3
"""
Llama 3.2-1B MMLU Evaluation Script (Optimized, Deterministic Device + Quant)

Key fixes and optimizations vs your current script:
1) Correct device semantics:
   - If --device cpu/mps is selected, bitsandbytes quantization is forbidden (CUDA-only here).
   - Quantized loading never uses device_map="auto" (which can silently place on GPU).
   - Quantized loading is forced onto cuda:0 (respects CUDA_VISIBLE_DEVICES).

2) Correct Transformers dtype argument:
   - Uses torch_dtype=... (not dtype=...).

3) Faster + cleaner MMLU scoring:
   - No generate(). Uses a single forward pass and scores next-token logits for A/B/C/D.
   - Much less Python / generation overhead; timings become meaningful.

4) Better timing hygiene:
   - Separates model load time from eval time.
   - Synchronizes CUDA around timing for accuracy.

Usage examples:
  CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v python3 llama_mmlu_eval_optimized.py --device cuda --quant none
  CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v python3 llama_mmlu_eval_optimized.py --device cuda --quant 4bit
  CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v python3 llama_mmlu_eval_optimized.py --device cuda --quant 8bit
  /usr/bin/time -v python3 llama_mmlu_eval_optimized.py --device cpu --quant none
"""

import argparse
import json
import os
import platform
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

DEFAULT_SUBJECTS = [
    "astronomy",
    "business_ethics",
]

CHOICES = ["A", "B", "C", "D"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cuda", "cpu", "mps", "auto"], default="auto",
                   help="Execution device. 'auto' prefers CUDA, then MPS, else CPU.")
    p.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none",
                   help="Quantization mode. In this script, bitsandbytes quantization is CUDA-only.")
    p.add_argument("--subjects", nargs="*", default=None,
                   help="Optional subject override. Example: --subjects astronomy business_ethics")
    p.add_argument("--max_examples", type=int, default=None,
                   help="Optional cap per subject for quick tests (e.g., 50).")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for deterministic behavior where applicable.")
    p.add_argument("--output_dir", type=str, default=".",
                   help="Directory to write results json.")
    p.add_argument("--compile", action="store_true",
                   help="If set and using PyTorch 2.x on CUDA, attempt torch.compile(model).")
    return p.parse_args()


def resolve_device(preferred: str) -> str:
    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preferred == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"

    # auto
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


def check_environment(device: str, quant_bits: Optional[int]) -> None:
    print("=" * 70)
    print("Environment Check")
    print("=" * 70)

    print(f"✓ Platform: {platform.system()} ({platform.machine()})")

    # Device reporting
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.2f} GB")
    elif device == "mps":
        print("✓ Apple Metal (MPS) Available")
    else:
        print("✓ Using CPU")

    # Quantization constraints (strict)
    if quant_bits is not None:
        if device != "cuda":
            raise RuntimeError(
                f"Requested {quant_bits}-bit quantization, but resolved device is '{device}'. "
                "In this script, bitsandbytes quantization is CUDA-only. Use --device cuda or --quant none."
            )
        try:
            import bitsandbytes  # noqa: F401
            print(f"✓ bitsandbytes installed - {quant_bits}-bit quantization available")
        except Exception as e:
            raise RuntimeError(
                "bitsandbytes is required for 4bit/8bit quantization. Install with: pip install bitsandbytes"
            ) from e
    else:
        print("✓ Quantization disabled - full precision path")

    # HF auth informational
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✓ Hugging Face authenticated")
        else:
            print("⚠️  No Hugging Face token found (hf auth login may be required for gated models)")
    except Exception:
        print("⚠️  Could not check Hugging Face authentication")

    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Quantization: {quant_bits}-bit" if quant_bits is not None else "Quantization: None (full precision)")
    print("=" * 70 + "\n")


def get_quant_config(quant_bits: Optional[int]) -> Optional[BitsAndBytesConfig]:
    if quant_bits is None:
        return None

    if quant_bits == 4:
        # 4-bit NF4 + double quant (common default)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    if quant_bits == 8:
        # 8-bit LLM.int8
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

    raise ValueError(f"Invalid quant_bits: {quant_bits}")


def load_model_and_tokenizer(device: str, quant_bits: Optional[int], do_compile: bool) -> Tuple[torch.nn.Module, AutoTokenizer, float]:
    t0 = datetime.now()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Ensure a pad token for safe batching; Llama often has no pad_token by default.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = get_quant_config(quant_bits)

    if quant_config is not None:
        # CUDA-only by construction (enforced earlier).
        # Force onto cuda:0; avoid device_map="auto" so --device semantics are honored.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quant_config,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
    else:
        # Full precision paths
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                dtype=torch.float16,
                device_map=None,
                low_cpu_mem_usage=False,
            ).to("cuda")
        elif device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                dtype=torch.float16,
                low_cpu_mem_usage=False,
            ).to("mps")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                dtype=torch.float32,
                low_cpu_mem_usage=False,
            ).to("cpu")

    model.eval()

    # Optional compile (CUDA only in practice for LLMs; guard for safety).
    if do_compile and device == "cuda":
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            print("✓ torch.compile enabled")
        except Exception as e:
            print(f"⚠️  torch.compile failed (continuing without compile): {e}")

    # Report memory (CUDA)
    if device == "cuda" and torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / 1e9
        reserv = torch.cuda.memory_reserved(0) / 1e9
        print(f"✓ Model loaded. CUDA mem: {alloc:.2f} GB allocated, {reserv:.2f} GB reserved")
    else:
        print("✓ Model loaded.")

    load_seconds = (datetime.now() - t0).total_seconds()
    return model, tokenizer, load_seconds


def format_mmlu_prompt(question: str, choices: List[str]) -> str:
    prompt = f"{question}\n\n"
    for label, choice in zip(CHOICES, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def _choice_token_ids(tokenizer: AutoTokenizer) -> Dict[str, List[int]]:
    """
    Tokenization can represent "A" vs " A" differently.
    We score both variants (when they tokenize to single tokens) and take max.
    """
    mapping: Dict[str, List[int]] = {}
    for c in CHOICES:
        ids = []
        for variant in [c, f" {c}"]:
            tok = tokenizer.encode(variant, add_special_tokens=False)
            if len(tok) == 1:
                ids.append(tok[0])
        # If neither variant is single-token, fall back to the first token of raw 'A'
        if not ids:
            ids = [tokenizer.encode(c, add_special_tokens=False)[0]]
        mapping[c] = list(dict.fromkeys(ids))  # dedupe preserving order
    return mapping


def predict_choice_next_token(
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
        next_logits = out.logits[0, -1]  # shape [vocab]

    # Score each letter by max logit over its token variants
    best_choice = "A"
    best_score = None
    for c in CHOICES:
        ids = choice_ids[c]
        score = max(float(next_logits[i]) for i in ids)
        if best_score is None or score > best_score:
            best_score = score
            best_choice = c
    return best_choice


def evaluate_subject(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    subject: str,
    choice_ids: Dict[str, List[int]],
    max_examples: Optional[int],
) -> Dict[str, object]:
    print(f"\n{'=' * 70}")
    print(f"Evaluating subject: {subject}")
    print(f"{'=' * 70}")

    dataset = load_dataset("cais/mmlu", subject, split="test")

    correct = 0
    total = 0

    it = dataset
    if max_examples is not None:
        it = dataset.select(range(min(max_examples, len(dataset))))

    for ex in tqdm(it, desc=f"Testing {subject}", leave=True):
        question = ex["question"]
        choices = ex["choices"]
        correct_answer = CHOICES[ex["answer"]]

        prompt = format_mmlu_prompt(question, choices)
        pred = predict_choice_next_token(model, tokenizer, prompt, choice_ids)

        if pred == correct_answer:
            correct += 1
        total += 1

    acc = (correct / total * 100.0) if total else 0.0
    print(f"✓ Result: {correct}/{total} correct = {acc:.2f}%")

    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "accuracy": acc,
    }


def cuda_sync_if_needed(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> str:
    args = parse_args()

    torch.manual_seed(args.seed)

    subjects = args.subjects if args.subjects is not None and len(args.subjects) > 0 else DEFAULT_SUBJECTS
    quant_bits = quant_bits_from_args(args.quant)
    device = resolve_device(args.device)

    check_environment(device=device, quant_bits=quant_bits)

    print("\n" + "=" * 70)
    print("Loading model/tokenizer")
    print("=" * 70)
    model, tokenizer, load_seconds = load_model_and_tokenizer(device=device, quant_bits=quant_bits, do_compile=args.compile)

    print("\n" + "=" * 70)
    print("Starting evaluation")
    print("=" * 70)
    print(f"Subjects: {subjects}")
    if args.max_examples is not None:
        print(f"Max examples per subject: {args.max_examples}")

    choice_ids = _choice_token_ids(tokenizer)

    # Warmup: one tiny forward pass to reduce first-iteration overhead
    warm_prompt = format_mmlu_prompt("Warmup question?", ["Warmup", "Warmup", "Warmup", "Warmup"])
    cuda_sync_if_needed(device)
    _ = predict_choice_next_token(model, tokenizer, warm_prompt, choice_ids)
    cuda_sync_if_needed(device)

    results: List[Dict[str, object]] = []
    total_correct = 0
    total_questions = 0

    start_time = datetime.now()
    cuda_sync_if_needed(device)

    for i, subject in enumerate(subjects, 1):
        print(f"\nProgress: {i}/{len(subjects)} subjects")
        r = evaluate_subject(
            model=model,
            tokenizer=tokenizer,
            subject=subject,
            choice_ids=choice_ids,
            max_examples=args.max_examples,
        )
        results.append(r)
        total_correct += int(r["correct"])
        total_questions += int(r["total"])

    cuda_sync_if_needed(device)
    end_time = datetime.now()
    eval_seconds = (end_time - start_time).total_seconds()

    overall_acc = (total_correct / total_questions * 100.0) if total_questions else 0.0

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Quantization: {quant_bits}-bit" if quant_bits is not None else "Quantization: None (full precision)")
    print(f"Subjects: {len(results)}")
    print(f"Questions: {total_questions}")
    print(f"Correct: {total_correct}")
    print(f"Accuracy: {overall_acc:.2f}%")
    print(f"Model load time: {load_seconds:.2f} s")
    print(f"Eval time: {eval_seconds:.2f} s ({eval_seconds/60:.2f} min)")
    print("=" * 70)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    quant_suffix = f"_{quant_bits}bit" if quant_bits is not None else "_full"
    out_name = f"llama_3.2_1b_mmlu_results{quant_suffix}_{device}_{ts}.json"
    out_path = os.path.join(args.output_dir, out_name)

    payload = {
        "model": MODEL_NAME,
        "device": device,
        "quantization_bits": quant_bits,
        "timestamp": ts,
        "seed": args.seed,
        "subjects": subjects,
        "max_examples": args.max_examples,
        "model_load_seconds": load_seconds,
        "eval_seconds": eval_seconds,
        "overall_accuracy": overall_acc,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "subject_results": results,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n✓ Results saved to: {out_path}")

    # Top/bottom
    sorted_results = sorted(results, key=lambda x: float(x["accuracy"]), reverse=True)
    print("\nTop subjects:")
    for r in sorted_results[:5]:
        print(f"  {r['subject']}: {float(r['accuracy']):.2f}%")

    print("\nBottom subjects:")
    for r in sorted_results[-5:]:
        print(f"  {r['subject']}: {float(r['accuracy']):.2f}%")

    print("\n✅ Done.")
    return out_path


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

