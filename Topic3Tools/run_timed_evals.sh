#!/usr/bin/env bash
# Run llama_mmlu_eval1.py and llama_mmlu_eval2.py with real clock-time measurement.
# Usage: ./run_timed_evals.sh   OR   bash run_timed_evals.sh
#
# Standalone function (paste into shell to use):
#   timed_mmlu() { time python "$1"; }
# Then:
#   cd /mnt/sdb/arafat/agentic_ai/Topic3Tools
#   timed_mmlu llama_mmlu_eval1.py    # astronomy
#   timed_mmlu llama_mmlu_eval2.py    # business_ethics
#
# Or one-liners (real = wall-clock elapsed):
#   time python llama_mmlu_eval1.py
#   time python llama_mmlu_eval2.py

timed_mmlu() {
  echo ">>> Running: $1"
  time python "$1"
  echo ""
}

cd "$(dirname "$0")"

echo "========== eval1 (astronomy) =========="
timed_mmlu llama_mmlu_eval1.py

echo "========== eval2 (business_ethics) =========="
timed_mmlu llama_mmlu_eval2.py

echo "Done. Check 'real' in each block for wall-clock time."
