"""
Task 1: Ollama MMLU Timing - Sequential and Parallel

Runs llama_mmlu_eval1_ollama.py and llama_mmlu_eval2_ollama.py
with timing. Use the shell script for exact `time` output:
  ./run_ollama_timing.sh

Or run this script for Python-based timing.
"""

import subprocess
import sys
import time

def run_cmd(cmd):
    start = time.perf_counter()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.perf_counter() - start
    return result.returncode, elapsed

def main():
    print("Task 1: Ollama MMLU Eval Timing")
    print("="*60)
    print("Prerequisites: ollama serve, ollama pull llama3.2:1b")
    print("="*60)

    # Sequential
    print("\n--- Sequential ---")
    t0 = time.perf_counter()
    subprocess.run([sys.executable, "llama_mmlu_eval1_ollama.py"], cwd=".")
    subprocess.run([sys.executable, "llama_mmlu_eval2_ollama.py"], cwd=".")
    t1 = time.perf_counter()
    print(f"\nSequential total: {t1-t0:.1f}s")

    # Parallel
    print("\n--- Parallel ---")
    t0 = time.perf_counter()
    p1 = subprocess.Popen([sys.executable, "llama_mmlu_eval1_ollama.py"], cwd=".")
    p2 = subprocess.Popen([sys.executable, "llama_mmlu_eval2_ollama.py"], cwd=".")
    p1.wait()
    p2.wait()
    t1 = time.perf_counter()
    print(f"\nParallel total: {t1-t0:.1f}s")

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
