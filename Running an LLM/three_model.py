#!/usr/bin/env python3
"""
MMLU Evaluation Script (Optimized, Deterministic Device + Quant + Timings)

Adds:
- --model to select HF model from CLI
- Timing summary:
  - Real (wall) time: time.perf_counter()
  - CPU process time: time.process_time()
  - GPU kernel time (CUDA only): torch.cuda.Event timing

Run once per model (recommended for clean attribution).
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

DEFAULT_SUBJECTS = [
    "astronomy",
    "business_ethics",
]

CHOICES = ["A", "B", "C", "D"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=(
            "Hugging Face model id to evaluate. Examples:\n"
            "  meta-llama/Llama-3.2-1B-Instruct\n"
            "  allenai/OLMo-2-0425-1B-Instruct\n"
            "  Qwen/Qwen2.5-0.5B-Instruct"
        ),
    )
    p.add_argument(
        "--device",
        choices=["cuda", "cpu", "mps", "auto"],
        default="auto",
        help="Execution device. 'auto' prefers CUDA, then MPS, else CPU.",
    )
    p.add_argument(
        "--quant",
        choices=["none", "4bit", "8bit"],
        default="none",
        help="Quantization mode. In this script, bitsandbytes quantization is CUDA-only.",
    )
    p.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject override. Example: --subjects astronomy business_ethics ...",
    )
    p.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap per subject for quick tests (e.g., 50).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic behavior where applicable.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to write results json.",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="If set and using PyTorch 2.x on CUDA, attempt torch.compile(model).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print each question, model answer, and correctness.",
    )
    p.add_argument(
        "--save_question_data",
        action="store_true",
        help="Save per-question results for mistake pattern analysis.",
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


def now_wall_s() -> float:
    return time.perf_counter()


def now_cpu_s() -> float:
    return time.process_time()


def cuda_sync_if_needed(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_event_timer_start(device: str):
    """
    Start CUDA-event timing. Returns (start_event, end_event) or None if not on CUDA.
    """
    if device == "cuda" and torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return (start, end)
    return None


def cuda_event_timer_stop(device: str, evt_pair) -> float:
    """
    Stop CUDA-event timing and return elapsed GPU time in seconds (kernel time).
    Returns 0.0 if not on CUDA.
    """
    if not evt_pair:
        return 0.0
    start, end = evt_pair
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    return float(ms) / 1000.0


def check_environment(model_name: str, device: str, quant_bits: Optional[int]) -> None:
    print("=" * 70)
    print("Environment Check")
    print("=" * 70)

    print(f"✓ Platform: {platform.system()} ({platform.machine()})")

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.2f} GB")
    elif device == "mps":
        print("✓ Apple Metal (MPS) Available")
    else:
        print("✓ Using CPU")

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
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Quantization: {quant_bits}-bit" if quant_bits is not None else "Quantization: None (full precision)")
    print("=" * 70 + "\n")


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


def load_model_and_tokenizer(
    model_name: str,
    device: str,
    quant_bits: Optional[int],
    do_compile: bool,
) -> Tuple[torch.nn.Module, AutoTokenizer, float]:
    t0 = datetime.now()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = get_quant_config(quant_bits)

    if quant_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
    else:
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map=None,
                low_cpu_mem_usage=False,
            ).to("cuda")
        elif device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                low_cpu_mem_usage=False,
            ).to("mps")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
                low_cpu_mem_usage=False,
            ).to("cpu")

    model.eval()

    if do_compile and device == "cuda":
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            print("✓ torch.compile enabled")
        except Exception as e:
            print(f"⚠️  torch.compile failed (continuing without compile): {e}")

    if device == "cuda" and torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / 1e9
        reserv = torch.cuda.memory_reserved(0) / 1e9
        print(f"✓ Model loaded. CUDA mem: {alloc:.2f} GB allocated, {reserv:.2f} GB reserved")
    else:
        print("✓ Model loaded.")

    load_seconds = (datetime.now() - t0).total_seconds()
    return model, tokenizer, load_seconds


def create_question_hash(subject: str, question: str, choices: List[str]) -> str:
    """Create a unique hash for a question to identify it across runs."""
    content = f"{subject}|||{question}|||{','.join(sorted(choices))}"
    return hashlib.md5(content.encode()).hexdigest()


def format_mmlu_prompt(question: str, choices: List[str]) -> str:
    prompt = f"{question}\n\n"
    for label, choice in zip(CHOICES, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def _choice_token_ids(tokenizer: AutoTokenizer) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    for c in CHOICES:
        ids: List[int] = []
        for variant in [c, f" {c}"]:
            tok = tokenizer.encode(variant, add_special_tokens=False)
            if len(tok) == 1:
                ids.append(tok[0])
        if not ids:
            ids = [tokenizer.encode(c, add_special_tokens=False)[0]]
        mapping[c] = list(dict.fromkeys(ids))
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
        next_logits = out.logits[0, -1]

    best_choice = "A"
    best_score: Optional[float] = None
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
    verbose: bool = False,
    save_questions: bool = False,
) -> Tuple[Dict[str, object], List[Dict]]:
    print(f"\n{'=' * 70}")
    print(f"Evaluating subject: {subject}")
    print(f"{'=' * 70}")

    dataset = load_dataset("cais/mmlu", subject, split="test")

    correct = 0
    total = 0
    question_data = []

    it = dataset
    if max_examples is not None:
        it = dataset.select(range(min(max_examples, len(dataset))))

    for idx, ex in enumerate(tqdm(it, desc=f"Testing {subject}", leave=True)):
        question = ex["question"]
        choices = ex["choices"]
        correct_answer = CHOICES[ex["answer"]]

        prompt = format_mmlu_prompt(question, choices)
        pred = predict_choice_next_token(model, tokenizer, prompt, choice_ids)

        is_correct = pred == correct_answer
        if is_correct:
            correct += 1
        total += 1

        if save_questions:
            qhash = create_question_hash(subject, question, choices)
            question_data.append({
                "subject": subject,
                "question_index": idx,
                "question": question,
                "choices": choices,
                "correct_answer": correct_answer,
                "model_answer": pred,
                "correct": is_correct,
                "question_hash": qhash,
            })

        if verbose:
            print(f"\nQuestion: {question}")
            for label, choice in zip(CHOICES, choices):
                marker = "✓" if label == correct_answer else " "
                print(f"  {marker} {label}. {choice}")
            print(f"Model Answer: {pred}")
            print(f"Correct Answer: {correct_answer}")
            print(f"Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
            print("-" * 70)

    acc = (correct / total * 100.0) if total else 0.0
    print(f"✓ Result: {correct}/{total} correct = {acc:.2f}%")

    result = {"subject": subject, "correct": correct, "total": total, "accuracy": acc}
    return result, question_data


def main() -> str:
    args = parse_args()
    torch.manual_seed(args.seed)

    model_name = args.model
    subjects = args.subjects if args.subjects and len(args.subjects) > 0 else DEFAULT_SUBJECTS
    quant_bits = quant_bits_from_args(args.quant)
    device = resolve_device(args.device)

    check_environment(model_name=model_name, device=device, quant_bits=quant_bits)

    print("\n" + "=" * 70)
    print("Loading model/tokenizer")
    print("=" * 70)

    load_wall_t0 = now_wall_s()
    load_cpu_t0 = now_cpu_s()
    model, tokenizer, load_seconds = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        quant_bits=quant_bits,
        do_compile=args.compile,
    )
    load_wall_s = now_wall_s() - load_wall_t0
    load_cpu_s = now_cpu_s() - load_cpu_t0

    print("\n" + "=" * 70)
    print("Starting evaluation")
    print("=" * 70)
    print(f"Subjects: {subjects}")
    if args.max_examples is not None:
        print(f"Max examples per subject: {args.max_examples}")

    choice_ids = _choice_token_ids(tokenizer)

    # Warmup
    warm_prompt = format_mmlu_prompt("Warmup question?", ["Warmup", "Warmup", "Warmup", "Warmup"])
    cuda_sync_if_needed(device)
    _ = predict_choice_next_token(model, tokenizer, warm_prompt, choice_ids)
    cuda_sync_if_needed(device)

    results: List[Dict[str, object]] = []
    total_correct = 0
    total_questions = 0

    eval_wall_t0 = now_wall_s()
    eval_cpu_t0 = now_cpu_s()
    cuda_sync_if_needed(device)
    eval_gpu_evt = cuda_event_timer_start(device)

    all_question_data = []
    for i, subject in enumerate(subjects, 1):
        print(f"\nProgress: {i}/{len(subjects)} subjects")
        r, qdata = evaluate_subject(
            model=model,
            tokenizer=tokenizer,
            subject=subject,
            choice_ids=choice_ids,
            max_examples=args.max_examples,
            verbose=args.verbose,
            save_questions=args.save_question_data,
        )
        results.append(r)
        total_correct += int(r["correct"])
        total_questions += int(r["total"])
        if args.save_question_data:
            all_question_data.extend(qdata)

    cuda_sync_if_needed(device)
    eval_gpu_s = cuda_event_timer_stop(device, eval_gpu_evt)
    eval_wall_s = now_wall_s() - eval_wall_t0
    eval_cpu_s = now_cpu_s() - eval_cpu_t0

    overall_acc = (total_correct / total_questions * 100.0) if total_questions else 0.0

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Quantization: {quant_bits}-bit" if quant_bits is not None else "Quantization: None (full precision)")
    print(f"Subjects: {len(results)}")
    print(f"Questions: {total_questions}")
    print(f"Correct: {total_correct}")
    print(f"Accuracy: {overall_acc:.2f}%")
    print("")
    print("Timing (this run)")
    print(f"  Load wall time: {load_wall_s:.3f} s")
    print(f"  Load CPU time:  {load_cpu_s:.3f} s")
    print(f"  Eval wall time: {eval_wall_s:.3f} s ({eval_wall_s/60:.2f} min)")
    print(f"  Eval CPU time:  {eval_cpu_s:.3f} s")
    if device == "cuda" and torch.cuda.is_available():
        print(f"  Eval GPU time:  {eval_gpu_s:.3f} s (CUDA events)")
    else:
        print("  Eval GPU time:  0.000 s (not on CUDA)")
    print("=" * 70)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    quant_suffix = f"_{quant_bits}bit" if quant_bits is not None else "_full"
    safe_model = model_name.replace("/", "__")
    out_name = f"mmlu_results_{safe_model}{quant_suffix}_{device}_{ts}.json"
    out_path = os.path.join(args.output_dir, out_name)

    payload = {
        "model": model_name,
        "device": device,
        "quantization_bits": quant_bits,
        "timestamp": ts,
        "seed": args.seed,
        "subjects": subjects,
        "max_examples": args.max_examples,
        "model_load_seconds": load_seconds,
        "eval_seconds": eval_wall_s,
        "timing": {
            "load_wall_s": load_wall_s,
            "load_cpu_s": load_cpu_s,
            "eval_wall_s": eval_wall_s,
            "eval_cpu_s": eval_cpu_s,
            "eval_gpu_s": eval_gpu_s,
        },
        "overall_accuracy": overall_acc,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "subject_results": results,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n✓ Results saved to: {out_path}")

    # Save question-level data if requested
    if args.save_question_data and all_question_data:
        question_out_name = f"mmlu_results_{safe_model}{quant_suffix}_{device}_{ts}_questions.json"
        question_out_path = os.path.join(args.output_dir, question_out_name)
        question_payload = {
            "model": model_name,
            "device": device,
            "quantization_bits": quant_bits,
            "timestamp": ts,
            "seed": args.seed,
            "subjects": subjects,
            "max_examples": args.max_examples,
            "questions": all_question_data,
        }
        with open(question_out_path, "w") as f:
            json.dump(question_payload, f, indent=2)
        print(f"✓ Question-level data saved to: {question_out_path}")

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
