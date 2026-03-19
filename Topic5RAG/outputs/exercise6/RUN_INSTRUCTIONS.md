# Exercise 6 — How to Run

## Overview

Test how different phrasings of the **same question** affect retrieval.

**Underlying question:** Engine maintenance (or change to any corpus-relevant topic)

**Phrasings (5):**
- Formal: "What is the recommended maintenance schedule for the engine?"
- Casual: "How often should I service the engine?"
- Keywords: "engine maintenance intervals"
- Question form: "When do I need to check the engine?"
- Indirect: "Preventive maintenance requirements"

## Prerequisites

1. Open `manual_rag_pipeline_universal.ipynb`
2. Load any corpus (Model T, Learjet, EU AI Act, etc.) and run full pipeline

## Where to Run

**Notebook only** — Run the **Exercise 6** cells (after Exercise 5):

1. First cell: Runs retrieval for each phrasing, records top-5 chunks and scores, computes overlap, saves results
2. Second cell: Creates observations template

## Output Files

- `phrasing_results.md` — Top-5 chunks and scores per phrasing, overlap analysis
- `observations.md` — Template to fill in (best phrasing, keywords vs natural, query rewriting implications)

## After Running

1. Review `phrasing_results.md` — compare chunks and scores across phrasings
2. Fill in `observations.md` — which phrasing retrieved best? Keywords vs natural?
3. Update `notes/findings.md` with Exercise 6 summary
