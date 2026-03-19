# Exercise 7 — How to Run

## Overview

Test how **chunk overlap** affects retrieval of information that spans chunk boundaries.

**Setup:** chunk_size=512 constant, overlap = 0, 64, 128, 256

**Query:** "What is the complete procedure for draining and replacing the oil in the crankcase?" (answer may span boundaries)

## Prerequisites

1. Open `manual_rag_pipeline_universal.ipynb`
2. Load Model T corpus (`Corpora/NewModelT`) and run full pipeline (including `rebuild_pipeline` and `retrieve` definitions)
3. **GPU recommended** — each rebuild re-embeds; takes several minutes per config. Course recommends Colab with T4+.

## Where to Run

**Notebook only** — Run the **Exercise 7** cells (after Exercise 6):

1. First cell: Loops over overlap 0, 64, 128, 256; rebuilds pipeline, runs query, saves results
2. Second cell: Creates observations template

**Runtime:** ~10–20+ minutes total (4 rebuilds × embedding time)

## Output Files

- `overlap_results.md` — n_chunks, top-5 retrieved chunks (scores + text preview), answer for each overlap
- `observations.md` — Template (index size, similarity scores, retrieval quality, diminishing returns, chunking implications)

## After Running

1. Review `overlap_results.md` — does higher overlap improve answer completeness?
2. Fill in `observations.md` — cost (index size, redundancy)? Diminishing returns?
3. Update `notes/findings.md` with Exercise 7 summary
