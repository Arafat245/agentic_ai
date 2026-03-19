# Topic5RAG — Retrieval Augmented Generation (RAG)

This directory contains my team’s work for Topic 5: RAG (Retrieval Augmented Generation) in Agentic AI Spring 2026 (CS 6501, University of Virginia).

## Table of Contents

- [Course Reference & Repo](#course-reference--repo)
- [Team Members](#team-members)
- [Learning Goals](#learning-goals)
- [Setup (Exercise 0)](#setup-exercise-0)
- [What We Ran](#what-we-ran)
- [Exercises Completed](#exercises-completed)
  - [Exercise 1 — RAG vs No-RAG](#exercise-1--open-model-rag-vs-no-rag-comparison)
  - [Exercise 2 — Open Model vs Large Model](#exercise-2--open-model--rag-vs-large-model-no-rag)
  - [Exercise 3 — Local RAG vs Frontier](#exercise-3--local-rag-vs-frontier-chat-model)
  - [Exercise 4 — Effect of Top-K](#exercise-4--effect-of-top-k-retrieval-count)
  - [Exercise 5 — Unanswerable Questions](#exercise-5--handling-unanswerable-questions)
  - [Exercise 6 — Query Phrasing](#exercise-6--query-phrasing-sensitivity)
  - [Exercise 7 — Chunk Overlap](#exercise-7--chunk-overlap-experiment)
  - [Exercise 8 — Chunk Size](#exercise-8--chunk-size-experiment)
  - [Exercise 9 — Retrieval Score Analysis](#exercise-9--retrieval-score-analysis)
  - [Exercise 10 — Prompt Templates](#exercise-10--prompt-template-variations)
  - [Exercise 11 — Cross-Document Synthesis](#exercise-11--cross-document-synthesis)
- [Repository Organization](#repository-organization)
- [How To Run](#how-to-run)
- [Notes](#notes)

---

## Course Reference & Repo

Course reference:
- [Topic 5: RAG (course page)](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic5RAG/rag.html)

Repo folder:
- [Topic5RAG (GitHub)](https://github.com/Arafat245/agentic_ai/tree/main/Topic5RAG)

RAG pipeline notebook and corpora are provided by the course page.

---

## Team Members

List of team members are here:

- Arafat Rahman
- Md Sabbir Ahmed


## Learning Goals

This topic focuses on understanding when to use RAG, what each component of a RAG pipeline does, and how retrieval parameters (chunking, top-k, prompt templates) affect quality and latency.


## Setup (Exercise 0)

1) Download and run the provided notebook:
- manual_rag_pipeline_universal.ipynb (linked on the course page)

2) Download and unzip the corpora:
- Corpora.zip (linked on the course page)

### Exercise 0 Progress — Document Loading ✓

| Step | Status | Output |
|------|--------|--------|
| Environment | ✓ | CUDA GPU (NVIDIA RTX A6000, 51.0 GB), LOCAL |
| Document source | ✓ | `Corpora/NewModelT` |
| Documents loaded | ✓ | 2 files: ModelTNew.txt (545,492 chars), ModelTNew.pdf (469,891 chars) |
| Document inspection | ✓ | First doc: ModelTNew.txt — Ford service manual, clean embedded text |

**Next:** Run Stage 2 (Chunking) — cells 13–16 in the notebook.

Notes from course updates (important for getting good retrieval results):
- Re-download Corpora.zip and use the “NewModelT” files (cleaner text) rather than older low-quality OCR. Prefer the .txt versions.
- The Congressional Record corpus can have garbled tables after extraction; some questions that depend on tables may not work well without improved preprocessing.
- In general, use the provided .txt files rather than extracting embedded PDF text in-code if possible.


## What We Ran

We used the course’s “manual RAG pipeline” notebook as the baseline implementation and then ran the required exercises below, saving outputs (answers, retrieved chunks, scores, notes) into this repository for each team member.


## Exercises Completed

The course page defines a sequence of RAG experiments (Exercises 1–10+). This README covers the core required experiments and how we organized outputs.

For every experiment below, we saved:
- the query,
- the answer without RAG,
- the answer with RAG,
- the retrieved chunk(s),
- any relevant retrieval scores (when applicable),
- short notes comparing quality and hallucination behavior.


### Exercise 1 — Open Model: RAG vs No-RAG Comparison

**How to run:** Open the notebook, run all cells through the RAG pipeline (cells 1–32), then run the **Exercise 1** cells (Part A → Switch to CR → Part B → Part C). See `outputs/exercise1/RUN_INSTRUCTIONS.md` for details.

We compared answers from a small open model:
- without retrieval (no RAG),
- with retrieval augmentation (RAG pipeline),

using two corpora separately:
- Model T repair manual
- Congressional Record corpus

The course page provides example queries for both corpora. 

### Exercise 2 — Open Model + RAG vs Large Model (No RAG)

**How to run:** From `Topic5RAG/` directory: `export OPENAI_API_KEY='...'` then `python run_exercise2.py`. See `outputs/exercise2/RUN_INSTRUCTIONS.md` for details.

We wrote/adapted a small script to query GPT-4o Mini (no tools, single-turn questions) and compared its answers to the Exercise 1 queries.

We documented:
- hallucination avoidance relative to the small open model,
- which questions GPT-4o Mini got right without retrieval,
- how this relates to the corpora’s time period vs the model’s training cut-off.


### Exercise 3 — Local RAG vs Frontier Chat Model

**How to run:** Local RAG answers from Exercise 1 (`model_t_results.txt`). Query GPT-4 or Claude via web (no file upload), paste answers into `outputs/exercise3/frontier_answers_template.md`, fill in `observations.md`. See `outputs/exercise3/RUN_INSTRUCTIONS.md`.

We compared:
- Local: small open model + RAG (Model T corpus)
- Cloud: a frontier chat model via web interface (no file upload)

We documented:
- where the frontier model’s general knowledge succeeds,
- where local RAG provides more specific/grounded answers,
- cases where the frontier model seems to use live web search.


### Exercise 4 — Effect of Top-K Retrieval Count

**How to run:** In the notebook, load Model T corpus and run the **Exercise 4** cells. Saves to `outputs/exercise4/`. See `outputs/exercise4/RUN_INSTRUCTIONS.md`.

We varied the number of retrieved chunks:
k = 1, 3, 5, 10, 20

For each k, we ran the same 3–5 queries and recorded:
- answer quality (accuracy, completeness),
- latency,
- when additional context stopped helping or started hurting.

### Exercise 5 — Handling Unanswerable Questions

**How to run:** In the notebook, run the **Exercise 5** cells. Saves to `outputs/exercise5/`. See `outputs/exercise5/RUN_INSTRUCTIONS.md`.

We tested unanswerable questions such as:
- completely off-topic,
- related but not in corpus,
- false-premise questions.

We documented:
- whether the model admits it cannot answer,
- whether it hallucinates,
- whether irrelevant retrieved context makes hallucination more likely.

We also tested prompt-template changes such as:
“If the context doesn’t contain the answer, say ‘I cannot answer this from the available documents.’”


### Exercise 6 — Query Phrasing Sensitivity

**How to run:** In the notebook, run the **Exercise 6** cells. Saves to `outputs/exercise6/`. See `outputs/exercise6/RUN_INSTRUCTIONS.md`.

We selected one underlying question and rephrased it 5+ ways (formal, casual, keywords-only, etc.), then recorded:
- top retrieved chunks,
- similarity scores,
- overlap between retrieval sets,
- which phrasing retrieved best context.

This experiment is used to reason about query rewriting strategies.


### Exercise 7 — Chunk Overlap Experiment

**How to run:** In the notebook, run the **Exercise 7** cells. Re-chunks with overlap 0, 64, 128, 256 (chunk_size=512). Takes long (each rebuild re-embeds). See `outputs/exercise7/RUN_INSTRUCTIONS.md`.

We tested overlap values and documented:
- whether higher overlap improves retrieval of information spanning chunk boundaries,
- index size and redundant information cost,
- point of diminishing returns.


### Exercise 8 — Chunk Size Experiment

**How to run:** In the notebook, run the **Exercise 8** cells. Re-chunks with sizes 128, 256, 512, 1024, 2048. Takes long (each rebuild re-embeds). See `outputs/exercise8/RUN_INSTRUCTIONS.md`.

We test chunk size values and document:
- whether larger chunks improve retrieval of complete procedures,
- index size and context granularity trade-off,
- point of diminishing returns.


### Exercise 9 — Retrieval Score Analysis

**How to run:** In the notebook, run the **Exercise 9** cells. 10 queries, top-10 chunks each. No rebuild — fast. See `outputs/exercise9/RUN_INSTRUCTIONS.md`.

We analyze retrieval score distribution:
- per-query min/max/mean/std,
- which queries had best/worst retrieval,
- implications for retrieval confidence and thresholds.


### Exercise 10 — Prompt Template Variations

**How to run:** In the notebook, run the **Exercise 10** cells. 5 templates × 3 queries. See `outputs/exercise10/RUN_INSTRUCTIONS.md`.

We test prompt templates: minimal, strict grounding, citation, permissive, structured — and compare accuracy, hallucination, traceability, usability.


### Exercise 11 — Cross-Document Synthesis

**How to run:** In the notebook, run the **Exercise 11** cells. 4 synthesis queries × top_k (3, 5, 10). See `outputs/exercise11/RUN_INSTRUCTIONS.md`.

We test synthesis queries (maintenance tasks, compare procedures, tools for tune-up, safety warnings) and document: can the model combine info from multiple chunks? Does higher top_k improve synthesis? Missed info? Contradictions?


## Repository Organization

```
Topic5RAG/
├── manual_rag_pipeline_universal.ipynb   # Main RAG pipeline notebook
├── run_exercise2.py                      # GPT-4o Mini script for Exercise 2
├── README.md                             # This file
├── Corpora/                              # Document corpora (from Corpora.zip)
│   ├── NewModelT/                        # Model T manual (ModelTNew.txt, .pdf)
│   ├── Congressional_Record_Jan_2026/    # CR Jan 2026 (txt/)
│   ├── Learjet/                          # Learjet manuals (txt/)
│   ├── EU_AI_Act/                        # EU AI Act
│   └── ModelTService/                    # Older Model T OCR (pdf_embedded/, txt/)
├── outputs/                              # Exercise outputs
│   ├── exercise1/                        # RAG vs No-RAG (model_t_results.txt, congressional_record_results.txt)
│   ├── exercise2/                        # GPT-4o Mini (gpt4o_mini_results.md)
│   ├── exercise3/                        # Local RAG vs Frontier (frontier_answers_template.md, local_rag_summary.md)
│   ├── exercise4/                        # Top-K (topk_results.md, latency_summary.md)
│   ├── exercise5/                        # Unanswerable (unanswerable_results.md)
│   ├── exercise6/                        # Query phrasing (phrasing_results.md)
│   ├── exercise7/                        # Chunk overlap (overlap_results.md)
│   ├── exercise8/                        # Chunk size (chunk_size_results.md)
│   ├── exercise9/                        # Retrieval scores (retrieval_scores.md)
│   ├── exercise10/                       # Prompt templates (prompt_results.md)
│   └── exercise11/                       # Cross-document synthesis (synthesis_results.md)
└── notes/
    ├── findings.md                       # Short write-up across experiments
    └── pitfalls.md                       # Preprocessing issues, table failures, etc.
```

Each `outputs/exerciseN/` folder contains:
- `RUN_INSTRUCTIONS.md` — How to run the exercise
- `observations.md` — Observations and analysis
- Results files (e.g., `*_results.md`, `*.txt`)

Save query outputs, answers, retrieved chunks, and scores into the corresponding `outputs/exerciseN/` folder. Update `notes/findings.md` and `notes/pitfalls.md` as you run experiments.

We ensured each team member has a copy of results in their GitHub, per the assignment instructions.

## How To Run

Option A (recommended by course): run on Colab, especially for long-running re-indexing experiments.

Option B (local): run the notebook locally.

Typical workflow:
1) Open the notebook.
2) Point the ingestion step to the desired .txt corpus.
3) Build the index (embedding + chunking).
4) Run queries with and without retrieval augmentation.
5) Save outputs into outputs/ and summarize observations in notes/.


## Notes

Preprocessing quality is decisive for retrieval quality, especially with multi-column PDFs and tables. The course page explicitly notes this limitation and recommends using the provided text corpora (and the cleaner Model T text).
