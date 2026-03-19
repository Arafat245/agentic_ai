# Exercise 3 — How to Run

## Overview

Compare **local RAG** (Qwen 2.5 1.5B + Model T corpus) vs **frontier chat model** (GPT-4, Claude, etc.) via web interface — **no file upload**, no RAG.

## Prerequisites

- **Local RAG answers** — From Exercise 1: `outputs/exercise1/model_t_results.txt` (RAG sections for the 4 Model T queries)
- **Frontier model access** — ChatGPT (GPT-4) or Claude via web: https://chat.openai.com or https://claude.ai

## Queries (Model T only — 4 questions)

1. How do I adjust the carburetor on a Model T?
2. What is the correct spark plug gap for a Model T Ford?
3. How do I fix a slipping transmission band?
4. What oil should I use in a Model T engine?

## Steps

### Step 1: Local RAG answers (already done)

Local RAG answers are in `outputs/exercise1/model_t_results.txt`. If you need to re-run:

1. Open `manual_rag_pipeline_universal.ipynb`
2. Load Model T corpus (`Corpora/NewModelT`), run pipeline
3. Run Exercise 1 Part A (Model T queries)

### Step 2: Query frontier model (manual)

1. Go to https://chat.openai.com (GPT-4) or https://claude.ai
2. **Do NOT upload any files.** Start a new chat.
3. Ask each of the 4 Model T queries above, one per message (or in one message).
4. Copy each answer into `frontier_answers_template.md`

### Step 3: Save frontier answers

Edit `frontier_answers_template.md` and paste the frontier model's responses under each query.

### Step 4: Compare and document

Fill in `observations.md`:
- Where does the frontier model's general knowledge succeed?
- Where does local RAG provide more accurate, specific answers?
- When did the frontier model appear to use live web search?

## Output Files

- `local_rag_summary.md` — Extracted RAG answers for quick reference (or use `../exercise1/model_t_results.txt`)
- `frontier_answers_template.md` — Paste frontier model answers here
- `observations.md` — Comparison and findings
