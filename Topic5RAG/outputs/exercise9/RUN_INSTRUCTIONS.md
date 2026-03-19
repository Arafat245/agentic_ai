# Exercise 9 — How to Run

## Overview

Analyze **retrieval score distribution** across 10 queries with top-10 chunks each.

**Setup:** 10 Model T queries, top_k=10, no pipeline rebuild

## Prerequisites

1. Open `manual_rag_pipeline_universal.ipynb`
2. Load Model T corpus (`Corpora/NewModelT`) and run full pipeline (including `retrieve` definition)

## Where to Run

**Notebook only** — Run the **Exercise 9** cells (after Exercise 8):

1. First cell: Runs 10 queries, retrieves top-10 chunks each, computes score stats, saves results
2. Second cell: Creates observations template

**Runtime:** ~1–2 minutes (no re-embedding; retrieval only)

## Output Files

- `retrieval_scores.md` — Per-query top-10 scores, min/max/mean/std, overall distribution, summary table
- `observations.md` — Template (score distribution, highest/lowest queries, spread, implications)

## After Running

1. Review `retrieval_scores.md` — which queries had best/worst retrieval?
2. Fill in `observations.md` — score spread, retrieval confidence, thresholds?
3. Update `notes/findings.md` with Exercise 9 summary
