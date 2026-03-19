#!/usr/bin/env bash
# Task 1: Ollama timing - sequential and parallel execution
# Prerequisites: ollama serve running, ollama pull llama3.2:1b
# Usage: ./run_ollama_timing.sh

cd "$(dirname "$0")"

echo "=============================================="
echo "Task 1: Ollama MMLU Eval - Sequential"
echo "=============================================="
time { python llama_mmlu_eval1_ollama.py ; python llama_mmlu_eval2_ollama.py ; }

echo ""
echo "=============================================="
echo "Task 1: Ollama MMLU Eval - Parallel"
echo "=============================================="
time { python llama_mmlu_eval1_ollama.py & python llama_mmlu_eval2_ollama.py & wait ; }

echo ""
echo "Done. Check 'real' time in each block."
