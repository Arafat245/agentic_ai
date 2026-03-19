# Exercise 4 — How to Run

## Overview

Vary the number of retrieved chunks (top_k) and observe effect on answer quality and latency.

**k values:** 1, 3, 5, 10, 20  
**Queries:** 4 Model T queries (carburetor, spark plug, transmission, engine oil)

## Prerequisites

1. Open `manual_rag_pipeline_universal.ipynb`
2. Load Model T corpus (`Corpora/NewModelT`), run full pipeline (chunk, embed, index, LLM)
3. Run through Exercise 1 Part A or ensure index is built

## Where to Run

**Notebook only** — Run the **Exercise 4** cells (after Exercise 3):

1. First cell: Runs all 4 queries for each k, measures latency, saves results
2. Second cell: Creates observations template

## Output Files

- `topk_results.md` — Full answers for each k and query
- `latency_summary.md` — Latency table (seconds per query per k)
- `observations.md` — Template to fill in (quality, when context helps/hurts)

## After Running

1. Review `topk_results.md` — compare answer quality across k
2. Review `latency_summary.md` — does latency increase with k?
3. Fill in `observations.md` — when does more context stop helping? When does it hurt?
4. Update `notes/findings.md` with Exercise 4 summary
