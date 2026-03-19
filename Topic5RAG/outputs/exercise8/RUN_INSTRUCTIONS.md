# Exercise 8 — How to Run

## Overview

Test how **chunk size** affects retrieval quality and answer completeness.

**Setup:** chunk_size = 128, 256, 512, 1024, 2048 (overlap = min(128, chunk_size//2))

**Query:** "What is the complete procedure for draining and replacing the oil in the crankcase?" (same as Exercise 7)

## Prerequisites

1. Open `manual_rag_pipeline_universal.ipynb`
2. Load Model T corpus (`Corpora/NewModelT`) and run full pipeline (including `rebuild_pipeline` and `retrieve` definitions)
3. **GPU recommended** — each rebuild re-embeds; takes several minutes per config. Course recommends Colab with T4+.

## Where to Run

**Notebook only** — Run the **Exercise 8** cells (after Exercise 7):

1. First cell: Loops over chunk sizes 128, 256, 512, 1024, 2048; rebuilds pipeline, runs query, saves results
2. Second cell: Creates observations template

**Runtime:** ~15–25+ minutes total (5 rebuilds × embedding time)

## Output Files

- `chunk_size_results.md` — n_chunks, top-5 retrieved chunks (scores + text preview), answer for each chunk size
- `observations.md` — Template (index size, similarity scores, retrieval quality, diminishing returns, chunking implications)

## After Running

1. Review `chunk_size_results.md` — does larger chunk size improve answer completeness?
2. Fill in `observations.md` — cost (index size)? Granularity vs. context?
3. Update `notes/findings.md` with Exercise 8 summary
