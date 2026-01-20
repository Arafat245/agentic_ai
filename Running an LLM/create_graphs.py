#!/usr/bin/env python3
"""
MMLU Results Analysis and Visualization Script

Analyzes results from multiple model evaluations and creates visualizations:
- Model comparison across subjects
- Mistake overlap analysis (which questions multiple models get wrong)
- Pattern analysis (random vs systematic mistakes)
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze MMLU evaluation results and create visualizations")
    p.add_argument(
        "--result_files",
        nargs="+",
        required=False,
        default=None,
        help="Paths to JSON result files from different models. Example: --result_files results1.json results2.json results3.json",
    )
    p.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="Directory containing result JSON files (alternative to --result_files)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save generated graphs",
    )
    p.add_argument(
        "--question_data_dir",
        type=str,
        default=None,
        help="Directory containing per-question JSON files (from --save_question_data in evaluation script)",
    )
    return p.parse_args()


def load_result_json(filepath: str) -> Dict:
    """Load a result JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def normalize_model_name(model: str) -> str:
    """Normalize model name to short format."""
    model_lower = model.lower()
    
    # Llama models
    if "llama-3.2-1b" in model_lower or "llama_3.2_1b" in model_lower or "llama__3.2-1b" in model_lower:
        return "Llama 3.2-1B"
    elif "llama-3.2-3b" in model_lower or "llama_3.2_3b" in model_lower:
        return "Llama 3.2-3B"
    elif "llama-3.1-8b" in model_lower or "llama_3.1_8b" in model_lower:
        return "Llama 3.1-8B"
    
    # OLMo models
    elif "olmo-2" in model_lower and "1b" in model_lower:
        return "OLMo 2 1B"
    elif "olmo-2" in model_lower and "7b" in model_lower:
        return "OLMo 2 7B"
    elif "olmo-3" in model_lower and "7b" in model_lower:
        return "OLMo 3 7B"
    
    # Qwen models
    elif "qwen2.5-0.5b" in model_lower or "qwen/qwen2.5-0.5b" in model_lower:
        return "Qwen 2.5 0.5B"
    elif "qwen2.5-1.5b" in model_lower or "qwen/qwen2.5-1.5b" in model_lower:
        return "Qwen 2.5 1.5B"
    elif "qwen2.5-3b" in model_lower or "qwen/qwen2.5-3b" in model_lower:
        return "Qwen 2.5 3B"
    elif "qwen2.5-7b" in model_lower or "qwen/qwen2.5-7b" in model_lower:
        return "Qwen 2.5 7B"
    
    # Other models
    elif "phi-2" in model_lower or "phi_2" in model_lower:
        return "Phi-2"
    elif "mistral-7b" in model_lower or "mistral_7b" in model_lower:
        return "Mistral 7B"
    elif "gemma" in model_lower and "2b" in model_lower:
        return "Gemma 2B"
    elif "tinyllama" in model_lower:
        return "TinyLlama"
    
    # Fallback: clean up common patterns
    cleaned = model.replace("/", " ").replace("__", " ").replace("_", " ")
    # Remove common prefixes
    cleaned = cleaned.replace("meta-llama ", "").replace("meta_llama ", "")
    cleaned = cleaned.replace("allenai/", "").replace("Qwen/", "")
    # Take first few words if too long
    words = cleaned.split()
    if len(words) > 4:
        cleaned = " ".join(words[:4])
    return cleaned


def get_model_name(result: Dict) -> str:
    """Extract a clean model name from result data."""
    model = result.get("model", "unknown")
    return normalize_model_name(model)


def create_question_hash(subject: str, question: str, choices: List[str]) -> str:
    """Create a unique hash for a question to identify it across runs."""
    content = f"{subject}|||{question}|||{','.join(sorted(choices))}"
    return hashlib.md5(content.encode()).hexdigest()


def plot_model_comparison(results_list: List[Tuple[str, Dict]], output_dir: str):
    """Create bar charts comparing models across subjects."""
    # Increase figure size significantly
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("Model Comparison Across Subjects", fontsize=22, fontweight="bold", y=0.995)
    
    # Set style for better readability
    plt.rcParams.update({'font.size': 14})
    
    # Collect data
    all_subjects = set()
    for _, result in results_list:
        for subj_result in result.get("subject_results", []):
            all_subjects.add(subj_result["subject"])
    all_subjects = sorted(all_subjects)
    
    model_names = [name for name, _ in results_list]
    subject_accuracies = {subj: [] for subj in all_subjects}
    
    for model_name, result in results_list:
        subj_dict = {r["subject"]: r["accuracy"] for r in result.get("subject_results", [])}
        for subj in all_subjects:
            subject_accuracies[subj].append(subj_dict.get(subj, 0.0))
    
    # Plot 1: Accuracy by subject (grouped bars)
    ax1 = axes[0, 0]
    x = np.arange(len(all_subjects))
    # Increased bar width to reduce spacing between groups
    width = 0.85 / len(model_names) if len(model_names) > 0 else 0.5
    
    for i, model_name in enumerate(model_names):
        accs = [subject_accuracies[subj][i] for subj in all_subjects]
        ax1.bar(x + i * width, accs, width, label=model_name, alpha=0.8)
    
    ax1.set_xlabel("Subject", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=16, fontweight="bold")
    ax1.set_title("Accuracy by Subject", fontsize=18, fontweight="bold", pad=10)
    
    if len(all_subjects) > 0:
        ax1.set_xticks(x + width * (len(model_names) - 1) / 2)
        # Rotate labels more and adjust spacing
        ax1.set_xticklabels(all_subjects, rotation=45, ha="right", fontsize=13)
    
    # Place legend inside plot area
    ax1.legend(loc='upper right', fontsize=13, framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 105)
    
    # Plot 2: Overall accuracy comparison
    ax2 = axes[0, 1]
    overall_accs = [result["overall_accuracy"] for _, result in results_list]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    # Reduced spacing between bars
    x_positions = np.arange(len(model_names)) * 1.0  # Normal spacing
    bar_width = 0.7  # Wider bars
    bars = ax2.bar(x_positions, overall_accs, width=bar_width, color=colors[:len(model_names)], alpha=0.8)
    
    ax2.set_ylabel("Overall Accuracy (%)", fontsize=16, fontweight="bold")
    ax2.set_title("Overall Accuracy Comparison", fontsize=18, fontweight="bold", pad=10)
    ax2.set_ylim(0, max(100, max(overall_accs) * 1.1))
    
    # Rotate x-axis labels vertically (90 degrees)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(model_names, rotation=90, ha="center", va="top", fontsize=14)
    
    # Add value labels on bars without % symbol
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=13, fontweight="bold")
    
    ax2.grid(axis="y", alpha=0.3, linestyle='--')
    
    # Plot 3: Subject accuracy heatmap
    ax3 = axes[1, 0]
    if len(all_subjects) > 0 and len(model_names) > 0:
        heatmap_data = np.array([subject_accuracies[subj] for subj in all_subjects])
        
        # Rotate model names vertically
        sns.heatmap(heatmap_data, 
                    xticklabels=model_names,
                    yticklabels=all_subjects,
                    annot=True, fmt=".1f", cmap="RdYlGn", 
                    vmin=0, vmax=100, ax=ax3,
                    cbar_kws={"label": "Accuracy (%)"},
                    annot_kws={"fontsize": 12})
        ax3.set_title("Accuracy Heatmap (Subject vs Model)", fontsize=18, fontweight="bold", pad=10)
        ax3.set_xlabel("Model", fontsize=16, fontweight="bold")
        ax3.set_ylabel("Subject", fontsize=16, fontweight="bold")
        ax3.tick_params(axis='x', labelsize=13, rotation=90)
        ax3.tick_params(axis='y', labelsize=13)
    
    # Plot 4: Error rate comparison
    ax4 = axes[1, 1]
    error_rates = [100 - result["overall_accuracy"] for _, result in results_list]
    error_colors = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    # Reduced spacing between bars
    x_positions = np.arange(len(model_names)) * 1.0  # Normal spacing
    bar_width = 0.7  # Wider bars
    bars = ax4.bar(x_positions, error_rates, width=bar_width, color=error_colors[:len(model_names)], alpha=0.8)
    
    ax4.set_ylabel("Error Rate (%)", fontsize=16, fontweight="bold")
    ax4.set_title("Overall Error Rate Comparison", fontsize=18, fontweight="bold", pad=10)
    ax4.set_ylim(0, max(100, max(error_rates) * 1.1))
    
    # Rotate x-axis labels vertically (90 degrees)
    ax4.set_xticks(x_positions)
    ax4.set_xticklabels(model_names, rotation=90, ha="center", va="top", fontsize=14)
    
    # Add value labels without % symbol
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=13, fontweight="bold")
    
    ax4.grid(axis="y", alpha=0.3, linestyle='--')
    
    # Adjust layout with reduced padding since legend is inside
    plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=2.0, w_pad=2.0)
    output_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"✓ Saved model comparison: {output_path}")
    plt.close()


def analyze_mistake_overlap(question_data_dir: str, output_dir: str):
    """Analyze which questions multiple models get wrong."""
    if not question_data_dir or not os.path.exists(question_data_dir):
        print("⚠️  Question-level data directory not found. Skipping mistake overlap analysis.")
        print("   Run evaluations with --save_question_data to enable this analysis.")
        return
    
    # Load question-level data files
    question_files = [f for f in os.listdir(question_data_dir) if f.endswith("_questions.json")]
    if not question_files:
        print("⚠️  No question-level JSON files found. Skipping mistake overlap analysis.")
        return
    
    # Map: question_hash -> {model: is_wrong}
    question_mistakes = defaultdict(dict)
    model_names = []
    
    for qfile in question_files:
        # Extract model name from filename
        model_name = qfile.replace("_questions.json", "").replace("mmlu_results_", "")
        # Remove timestamp and other suffixes
        parts = model_name.split("_")
        model_name = "_".join(parts[:-2])  # Remove last two parts (device and timestamp)
        # Normalize to short name format
        model_name = normalize_model_name(model_name)
        model_names.append(model_name)
        
        filepath = os.path.join(question_data_dir, qfile)
        with open(filepath, "r") as f:
            data = json.load(f)
        
        for entry in data.get("questions", []):
            qhash = entry["question_hash"]
            question_mistakes[qhash][model_name] = not entry["correct"]
    
    if not question_mistakes:
        print("⚠️  No question data found in files. Skipping mistake overlap analysis.")
        return
    
    # Analyze overlap
    mistake_counts = defaultdict(int)  # number_of_models_wrong -> count
    questions_by_overlap = defaultdict(list)
    
    for qhash, model_results in question_mistakes.items():
        wrong_count = sum(1 for is_wrong in model_results.values() if is_wrong)
        if wrong_count > 0:
            mistake_counts[wrong_count] += 1
            if wrong_count > 1:
                questions_by_overlap[wrong_count].append(qhash)
    
    # Plot 1: Distribution of mistake overlap (single plot)
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 7))
    plt.rcParams.update({'font.size': 14})
    
    overlap_levels = sorted(mistake_counts.keys())
    if not overlap_levels:
        print("⚠️  No mistakes found in question data.")
        plt.close()
        return
    
    counts = [mistake_counts[level] for level in overlap_levels]
    bars = ax1.bar([f"{l} model(s)" for l in overlap_levels], counts, 
                   color=plt.cm.Reds(np.linspace(0.4, 0.9, len(overlap_levels))),
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_xlabel("Number of Models That Got It Wrong", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Number of Questions", fontsize=16, fontweight="bold")
    ax1.set_title("Mistake Overlap: How Many Models Get Each Question Wrong?", 
                  fontsize=18, fontweight="bold", pad=10)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=14, fontweight="bold")
    
    ax1.grid(axis="y", alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(counts) * 1.15)
    
    plt.tight_layout()
    output_path1 = os.path.join(output_dir, "mistake_overlap_distribution.png")
    plt.savefig(output_path1, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"✓ Saved mistake overlap distribution: {output_path1}")
    plt.close()
    
    # Plot 2: Pairwise comparison (separate large figure)
    if len(model_names) >= 2:
        pairs = []
        for i, m1 in enumerate(model_names):
            for m2 in model_names[i+1:]:
                both_wrong = sum(1 for qhash, results in question_mistakes.items()
                               if results.get(m1, False) and results.get(m2, False))
                only_m1 = sum(1 for qhash, results in question_mistakes.items()
                            if results.get(m1, False) and not results.get(m2, False))
                only_m2 = sum(1 for qhash, results in question_mistakes.items()
                            if not results.get(m1, False) and results.get(m2, False))
                pairs.append((m1, m2, both_wrong, only_m1, only_m2))
        
        if pairs:
            # Split pairs into 5 groups
            num_pairs = len(pairs)
            pairs_per_section = (num_pairs + 4) // 5  # Divide into 5 roughly equal parts
            
            # Create figure with 5 subplots
            fig2, axes2 = plt.subplots(5, 1, figsize=(max(18, num_pairs * 1.0), 20))
            plt.rcParams.update({'font.size': 16})
            fig2.suptitle("Pairwise Mistake Overlap", fontsize=22, fontweight="bold", y=0.995)
            
            both_wrong_counts = [p[2] for p in pairs]
            only_first = [p[3] for p in pairs]
            only_second = [p[4] for p in pairs]
            
            # Calculate max value for consistent y-axis
            max_val = max(max(both_wrong_counts), max(only_first), max(only_second))
            
            # Split into 5 sections
            sections = [
                (0, pairs_per_section, "Top"),
                (pairs_per_section, min(2 * pairs_per_section, num_pairs), "Upper-Middle"),
                (2 * pairs_per_section, min(3 * pairs_per_section, num_pairs), "Middle"),
                (3 * pairs_per_section, min(4 * pairs_per_section, num_pairs), "Lower-Middle"),
                (4 * pairs_per_section, num_pairs, "Bottom")
            ]
            
            for section_idx, (start_idx, end_idx, section_name) in enumerate(sections):
                if start_idx >= end_idx:
                    axes2[section_idx].axis('off')
                    continue
                    
                ax = axes2[section_idx]
                section_pairs = pairs[start_idx:end_idx]
                section_both = both_wrong_counts[start_idx:end_idx]
                section_only_first = only_first[start_idx:end_idx]
                section_only_second = only_second[start_idx:end_idx]
                
                # Create labels with full model names
                pair_labels = []
                for p in section_pairs:
                    pair_labels.append(f"{p[0]}\nvs\n{p[1]}")
                
                # Reduced spacing between bar groups
                x = np.arange(len(section_pairs)) * 1.0
                width = 0.25  # Reduced bar width
                
                bars1 = ax.bar(x - width, section_both, width, label="Both wrong", 
                               color="#d62728", alpha=0.8, edgecolor='black', linewidth=0.5)
                bars2 = ax.bar(x, section_only_first, width, label="Only first wrong", 
                               color="#ff7f0e", alpha=0.8, edgecolor='black', linewidth=0.5)
                bars3 = ax.bar(x + width, section_only_second, width, label="Only second wrong", 
                               color="#9467bd", alpha=0.8, edgecolor='black', linewidth=0.5)
                
                ax.set_ylabel("Number of Questions", fontsize=18, fontweight="bold")
                ax.set_title(f"{section_name} Section (Pairs {start_idx+1}-{end_idx})", 
                            fontsize=20, fontweight="bold", pad=10)
                ax.set_xticks(x)
                # Rotate labels vertically with larger font
                ax.set_xticklabels(pair_labels, rotation=90, ha="center", va="top", fontsize=16)
                
                if section_idx == 0:
                    ax.legend(loc='upper right', fontsize=16, framealpha=0.9)
                
                ax.grid(axis="y", alpha=0.3, linestyle='--')
                ax.set_ylim(0, max_val * 1.15)
                
                # Add value labels on bars
                for bars_group in [bars1, bars2, bars3]:
                    for bar in bars_group:
                        height = bar.get_height()
                        if height > 0:  # Only label if value > 0
                            ax.text(bar.get_x() + bar.get_width()/2., height + max_val * 0.01,
                                    f'{int(height)}', ha='center', va='bottom', fontsize=15, fontweight="bold")
            
            plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=2.0)
            output_path2 = os.path.join(output_dir, "mistake_overlap_pairwise.png")
            plt.savefig(output_path2, dpi=300, bbox_inches="tight", facecolor='white')
            print(f"✓ Saved pairwise mistake overlap: {output_path2}")
            plt.close()
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("MISTAKE OVERLAP ANALYSIS")
    print("=" * 70)
    total_questions = len(question_mistakes)
    print(f"Total questions analyzed: {total_questions}")
    if len(model_names) > 0:
        print(f"Models analyzed: {', '.join(model_names)}")
    print(f"\nQuestions that {len(model_names)} models got wrong: {mistake_counts.get(len(model_names), 0)}")
    if len(model_names) > 1:
        print(f"Questions that {len(model_names)-1} models got wrong: {mistake_counts.get(len(model_names)-1, 0)}")
    print(f"Questions unique to one model: {mistake_counts.get(1, 0)}")
    print("=" * 70 + "\n")


def main():
    args = parse_args()
    
    # Collect result files
    result_files = []
    if args.result_files:
        result_files.extend(args.result_files)
    if args.result_dir:
        if os.path.exists(args.result_dir):
            result_files.extend([
                os.path.join(args.result_dir, f)
                for f in os.listdir(args.result_dir)
                if f.startswith("mmlu_results_") and f.endswith(".json") and not f.endswith("_questions.json")
            ])
        else:
            print(f"⚠️  Result directory not found: {args.result_dir}")
    
    if not result_files:
        print("❌ No result files found. Use --result_files or --result_dir")
        print("   Example: python create_graphs.py --result_dir . --question_data_dir .")
        return
    
    print(f"Loading {len(result_files)} result file(s)...")
    results_dict = {}  # Use dict to track most recent result per model
    
    for filepath in result_files:
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filepath}, skipping...")
            continue
        try:
            result = load_result_json(filepath)
            model_name = get_model_name(result)
            
            # Get file modification time as fallback
            file_mtime = os.path.getmtime(filepath)
            
            # Extract timestamp from filename or result for comparison
            timestamp = result.get("timestamp", "")
            if not timestamp:
                # Try to extract from filename
                filename = os.path.basename(filepath)
                parts = filename.split("_")
                if len(parts) >= 2:
                    timestamp = parts[-1].replace(".json", "")
            
            # Keep only the most recent result for each model
            if model_name not in results_dict:
                results_dict[model_name] = (timestamp, file_mtime, result, filepath)
                print(f"✓ Loaded: {model_name} ({filepath})")
            else:
                # Compare timestamps and keep the most recent
                existing_timestamp, existing_mtime, _, existing_filepath = results_dict[model_name]
                
                # Use timestamp comparison if both have timestamps, otherwise use file mtime
                should_replace = False
                if timestamp and existing_timestamp:
                    should_replace = timestamp > existing_timestamp
                else:
                    should_replace = file_mtime > existing_mtime
                
                if should_replace:
                    results_dict[model_name] = (timestamp, file_mtime, result, filepath)
                    print(f"✓ Updated: {model_name} (replaced older result from {existing_filepath})")
                    print(f"  New file: {filepath}")
                else:
                    print(f"⊘ Skipped: {model_name} (keeping newer result from {existing_filepath})")
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
    
    # Convert dict to list format
    results_list = [(model_name, result) for model_name, (_, _, result, _) in results_dict.items()]
    
    if not results_list:
        print("❌ No valid result files loaded.")
        return
    
    print(f"\n✓ Loaded {len(results_list)} unique model(s) after deduplication")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create visualizations
    print("\n" + "=" * 70)
    print("Creating visualizations...")
    print("=" * 70)
    
    plot_model_comparison(results_list, args.output_dir)
    
    # Analyze mistake overlap if question-level data is available
    if args.question_data_dir:
        analyze_mistake_overlap(args.question_data_dir, args.output_dir)
    else:
        print("\n💡 Tip: Use --question_data_dir to enable mistake overlap analysis")
        print("   First run evaluations with --save_question_data flag")
    
    print("\n✅ Analysis complete!")
    print(f"   Graphs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
