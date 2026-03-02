# Topic4Exploring — Exploring Agentic AI

This directory contains my work for Topic 4: Exploring in Agentic AI Spring 2026 (CS 6501, University of Virginia).

The goal of this topic is to explore agentic behaviors and system design tradeoffs by running structured experiments on agent graphs, memory, prompting, routing, and execution strategies, building on the previous topics (frameworks and tools).

Assignment reference:
[Topic 4: Exploring (course page)](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic4Exploring/exploring.html)

Repo folder:
[Topic4Exploring (GitHub)](https://github.com/Arafat245/agentic_ai/tree/main/Topic4Exploring)


## Overview

In this topic, I focus on experimentation and analysis rather than only “getting a working agent”.

Typical explorations include:

- How changes in graph structure affect behavior and failure modes.
- How history length and memory format influence responses.
- How routing decisions change tool usage patterns.
- Where parallelism is possible (and where it breaks correctness).
- How to log and interpret intermediate state transitions.
- How sensitive the system is to prompt phrasing and system instructions.


## Requirements

Install dependencies:

pip install -r requirements.txt

Depending on the scripts in this folder, you may also need:

- OPENAI_API_KEY exported (for OpenAI-backed runs).
- A local model server (e.g., Ollama) if any scripts use local inference.

Example OpenAI env setup:

export OPENAI_API_KEY="your_key_here"


## Files

This folder contains exploration scripts and notes. Each script is intended to be runnable independently and produce console output (and optionally a log / JSON / text artifact).

Common file patterns in this topic:

- Exploration scripts: explore*.py
- Notes / writeups: *.txt or *.md
- Output artifacts: *.json, *.csv, logs


## What Each Exploration Covers

The scripts in this directory explore different dimensions of an agentic system. The exact filenames may vary, but the intent is:

1) Performance and profiling
- Measure latency per node (LLM call vs tool call vs routing).
- Identify bottlenecks and evaluate optimizations.

2) Memory and history effects
- Compare “no memory” vs “short memory” vs “long memory”.
- Evaluate message formatting (system/user/assistant roles).
- Observe drift, contradictions, and context-overload behaviors.

3) Routing and control flow sensitivity
- Change conditional edges and observe behavior changes.
- Investigate loops, early-exit conditions, and failure recovery.

4) Parallelism experiments
- Try parallel tool calls or parallel model calls.
- Compare sequential vs parallel wall-clock time.
- Note correctness constraints (dependencies, shared state).

5) Prompt sensitivity and robustness
- Small prompt perturbations, different system prompts.
- Measure response stability and tool-use consistency.

6) Logging and interpretability
- Record intermediate state dictionaries.
- Save traces for later inspection and debugging.


## Running

From inside this directory:

python <script_name>.py

Examples:

python explore1_*.py
python explore2_*.py

Each script prints its own instructions and results to the terminal. Some scripts may prompt for input.


## Expected Outputs

Depending on the specific exploration:

- Console logs showing decisions (routes chosen, tool calls made).
- Timing summaries (per node, per turn, total runtime).
- Saved artifacts:
  - JSON logs of state transitions
  - CSV timing tables
  - Text summaries of findings


## Notes

- Topic 4 is exploratory: results may differ by model backend, temperature, and hardware.
- When recording results, include the model name, temperature, and environment details.
- Keep outputs (logs/tables) alongside scripts for portfolio submission.

This directory is intended to show not only working code, but reasoning about agent behavior under controlled variations.
