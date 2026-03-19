"""
Llama 3.2-1B MMLU Evaluation via Ollama (Topic: business_ethics)

Uses Ollama server instead of HuggingFace transformers.
Requires: ollama serve running, ollama pull llama3.2:1b
"""

import ollama
from datasets import load_dataset
import json
from tqdm.auto import tqdm
from datetime import datetime

MODEL_NAME = "llama3.2:1b"
MMLU_SUBJECTS = ["business_ethics"]


def format_mmlu_prompt(question, choices):
    """Format MMLU question as multiple choice"""
    choice_labels = ["A", "B", "C", "D"]
    prompt = f"{question}\n\n"
    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer with just the letter (A, B, C, or D):"
    return prompt


def get_model_prediction(prompt):
    """Get model's prediction via Ollama"""
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 5, "temperature": 0}
        )
        generated_text = response["message"]["content"].strip()
    except Exception as e:
        print(f"Ollama error: {e}")
        return "A"

    answer = generated_text[:1].upper() if generated_text else "A"
    if answer not in ["A", "B", "C", "D"]:
        for char in generated_text.upper():
            if char in ["A", "B", "C", "D"]:
                answer = char
                break
        else:
            answer = "A"
    return answer


def evaluate_subject(subject):
    """Evaluate model on a specific MMLU subject"""
    print(f"\n{'='*70}")
    print(f"Evaluating subject: {subject} (via Ollama)")
    print(f"{'='*70}")

    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"Error loading subject {subject}: {e}")
        return None

    correct = 0
    total = 0

    for example in tqdm(dataset, desc=f"Testing {subject}", leave=True):
        question = example["question"]
        choices = example["choices"]
        correct_answer_idx = example["answer"]
        correct_answer = ["A", "B", "C", "D"][correct_answer_idx]

        prompt = format_mmlu_prompt(question, choices)
        predicted_answer = get_model_prediction(prompt)

        if predicted_answer == correct_answer:
            correct += 1
        total += 1

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"Result: {correct}/{total} correct = {accuracy:.2f}%")

    return {"subject": subject, "correct": correct, "total": total, "accuracy": accuracy}


def main():
    print("\n" + "="*70)
    print("Llama 3.2-1B MMLU Evaluation via Ollama (business_ethics)")
    print("="*70 + "\n")

    results = []
    total_correct = 0
    total_questions = 0

    for subject in MMLU_SUBJECTS:
        result = evaluate_subject(subject)
        if result:
            results.append(result)
            total_correct += result["correct"]
            total_questions += result["total"]

    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Model: {MODEL_NAME} (Ollama)")
    print(f"Total Correct: {total_correct}/{total_questions} = {overall_accuracy:.2f}%")
    print("="*70)
    return results


if __name__ == "__main__":
    main()
