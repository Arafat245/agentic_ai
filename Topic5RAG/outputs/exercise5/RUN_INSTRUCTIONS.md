# Exercise 5 — How to Run

## Overview

Test how the RAG system handles questions it **cannot** answer from the corpus.

**Question types:**
- **Off-topic:** "What is the capital of France?"
- **Related but not in corpus:** "What's the horsepower of a 1925 Model T?"
- **False premise:** "Why does the manual recommend synthetic oil?" (it doesn't)

**Experiment:** Compare default prompt vs strict prompt: "If the context doesn't contain the answer, say 'I cannot answer this from the available documents.'"

## Prerequisites

1. Open `manual_rag_pipeline_universal.ipynb`
2. Load any corpus (Model T, Congressional Record, etc.) and run full pipeline

## Where to Run

**Notebook only** — Run the **Exercise 5** cells (after Exercise 4):

1. First cell: Runs 3 unanswerable questions with default and strict prompts, saves results
2. Second cell: Creates observations template

## Output Files

- `unanswerable_results.md` — Answers for each question (default + strict prompt)
- `observations.md` — Template to fill in (admits? hallucinates? does strict help?)

## After Running

1. Review `unanswerable_results.md` — does the model admit it doesn't know?
2. Fill in `observations.md` — does irrelevant context encourage hallucination? Does strict prompt help?
3. Update `notes/findings.md` with Exercise 5 summary
