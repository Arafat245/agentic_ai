# Exercise 11 — How to Run

## Overview

Test **cross-document synthesis** — questions that require combining information from multiple chunks.

**Queries:**
- What are ALL the maintenance tasks I need to do monthly?
- Compare the procedures for adjusting the carburetor vs adjusting the transmission band.
- What tools do I need for a complete tune-up?
- Summarize all safety warnings in the manual.

**Experiment:** top_k = 3, 5, 10 — does retrieving more chunks improve synthesis?

## Prerequisites

1. Open `manual_rag_pipeline_universal.ipynb`
2. Load corpus (Model T or any) and run full pipeline

## Where to Run

**Notebook only** — Run the **Exercise 11** cells (after Exercise 10):

1. First cell: Runs 4 synthesis queries with top_k=3, 5, 10 each, saves results
2. Second cell: Creates observations template

**Runtime:** ~5–15 minutes (12 RAG calls)

## Output Files

- `synthesis_results.md` — Answers for each query × top_k
- `observations.md` — Template (synthesis quality, top_k effect, missed info, contradictions)

## Document

- Can the model successfully combine information from multiple chunks?
- Does it miss information that wasn't retrieved?
- Does contradictory information in different chunks cause problems?

## After Running

1. Review `synthesis_results.md` — does higher top_k improve synthesis?
2. Fill in `observations.md` — synthesis quality, missed info, contradictions?
3. Update `notes/findings.md` with Exercise 11 summary
