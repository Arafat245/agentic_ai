# Exercise 10 — How to Run

## Overview

Test **prompt template variations** and compare answers: minimal, strict grounding, citation, permissive, structured.

**Setup:** 5 templates × 3 queries (carburetor, spark plug, engine oil)

## Prerequisites

1. Open `manual_rag_pipeline_universal.ipynb`
2. Load Model T corpus (`Corpora/NewModelT`) and run full pipeline (including `rag_query` with `prompt_template` support)

## Where to Run

**Notebook only** — Run the **Exercise 10** cells (after Exercise 9):

1. First cell: Runs 3 queries with each of 5 prompt templates, saves answers
2. Second cell: Creates observations template

**Runtime:** ~5–15 minutes (15 RAG calls total)

## Output Files

- `prompt_results.md` — Answers for each query × template combination
- `observations.md` — Template (accuracy, hallucination, citation, structured format, implications)

## After Running

1. Review `prompt_results.md` — which template produced best answers?
2. Fill in `observations.md` — accuracy, hallucination, citation, usability?
3. Update `notes/findings.md` with Exercise 10 summary
