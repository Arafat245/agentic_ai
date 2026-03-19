#!/usr/bin/env python3
"""
Cross-Dataset Comparison Graphs

Creates graphs comparing model accuracy across multiple benchmark datasets
(MMLU, ARC, etc.) for portfolio compliance: "comparing accuracy and performance
of several different models on several different benchmark datasets".

Output: PDF format for portfolio submission.
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create cross-dataset comparison graphs")
    p.add_argument(
        "--result_dir",
        type=str,
        default=".",
        help="Directory containing mmlu_results_*.json and arc_results_*.json",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="graphs",
        help="Directory to save PDF graphs",
    )
    p.add_argument(
        "--format",
        choices=["pdf", "png"],
        default="pdf",
        help="Output format (default: pdf for portfolio)",
    )
    return p.parse_args()


def normalize_model_name(model: str) -> str:
    """Short model name for display."""
    m = model.lower()
    if "llama-3.2-1b" in m or "llama__3.2-1b" in m:
        return "Llama 3.2-1B"
    if "olmo-2" in m and "1b" in m:
        return "OLMo 2 1B"
    if "qwen2.5-0.5b" in m:
        return "Qwen 2.5 0.5B"
    if "qwen2.5-1.5b" in m:
        return "Qwen 2.5 1.5B"
    if "qwen2.5-3b" in m:
        return "Qwen 2.5 3B"
    if "qwen2.5-7b" in m:
        return "Qwen 2.5 7B"
    if "phi-2" in m:
        return "Phi-2"
    # Fallback
    return model.replace("/", " ").replace("__", " ").split()[0] + " ..."


def load_all_results(result_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Load MMLU and ARC results.
    Returns: {model_short_name: {dataset_name: accuracy}}
    """
    data: Dict[str, Dict[str, float]] = defaultdict(dict)
    mmlu_ts: Dict[str, str] = {}  # model -> timestamp for keeping most recent

    if not os.path.exists(result_dir):
        return dict(data)

    for fname in os.listdir(result_dir):
        if not fname.endswith(".json") or "_questions.json" in fname:
            continue

        filepath = os.path.join(result_dir, fname)
        try:
            with open(filepath, "r") as f:
                r = json.load(f)
        except Exception:
            continue

        model = r.get("model", "unknown")
        model_short = normalize_model_name(model)
        acc = float(r.get("overall_accuracy", 0))
        ts = r.get("timestamp", "")

        if "mmlu_results" in fname or "llama_3.2_1b_mmlu" in fname:
            if model_short not in mmlu_ts or ts > mmlu_ts[model_short]:
                data[model_short]["MMLU"] = acc
                mmlu_ts[model_short] = ts
        elif "arc_results" in fname:
            dataset = r.get("dataset", "ARC")
            data[model_short][dataset] = acc

    return dict(data)


def plot_cross_dataset_comparison(
    model_data: Dict[str, Dict[str, float]],
    output_dir: str,
    fmt: str = "pdf",
) -> None:
    """Create grouped bar chart: models x datasets."""
    if not model_data:
        print("⚠️  No result data found. Run MMLU and ARC evaluations first.")
        return

    # Collect all datasets and models
    all_datasets = set()
    for per_model in model_data.values():
        all_datasets.update(k for k in per_model.keys() if not k.startswith("_"))
    all_datasets = sorted(all_datasets)

    models = sorted(model_data.keys())

    if not models or not all_datasets:
        print("⚠️  No valid model/dataset combinations.")
        return

    # Build matrix: rows=datasets, cols=models
    matrix = []
    for ds in all_datasets:
        row = [model_data[m].get(ds, 0.0) for m in models]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(models))
    width = 0.8 / len(all_datasets)
    offset = (len(all_datasets) - 1) * width / 2

    colors = plt.cm.Set2(np.linspace(0, 1, len(all_datasets)))

    for i, ds in enumerate(all_datasets):
        vals = matrix[i]
        bars = ax.bar(
            x - offset + i * width,
            vals,
            width,
            label=ds,
            color=colors[i],
            alpha=0.85,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.5,
                    f"{h:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    rotation=0,
                )

    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Model Performance Across Benchmark Datasets\n(MMLU + ARC)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"cross_dataset_comparison.{fmt}")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", format=fmt)
    plt.close()
    print(f"✓ Saved cross-dataset comparison: {out_path}")


def main():
    args = parse_args()
    model_data = load_all_results(args.result_dir)

    if not model_data:
        print("❌ No MMLU or ARC result files found in:", args.result_dir)
        print("   Run: python3 llama_mmlu_eval_optimized.py ... (MMLU)")
        print("   Run: python3 arc_eval.py ... (ARC)")
        return

    print(f"Loaded results for {len(model_data)} model(s)")
    for m, d in model_data.items():
        print(f"  {m}: {list(d.keys())}")

    os.makedirs(args.output_dir, exist_ok=True)
    plot_cross_dataset_comparison(model_data, args.output_dir, fmt=args.format)
    print("\n✅ Cross-dataset graphs complete!")


if __name__ == "__main__":
    main()
