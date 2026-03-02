# Topic5RAG — Retrieval Augmented Generation (RAG)

This directory contains my team’s work for Topic 5: RAG (Retrieval Augmented Generation) in Agentic AI Spring 2026 (CS 6501, University of Virginia).

Course reference:
- [Topic 5: RAG (course page)](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic5RAG/rag.html)

Repo folder:
- [Topic5RAG (GitHub)](https://github.com/Arafat245/agentic_ai/tree/main/Topic5RAG)

RAG pipeline notebook and corpora are provided by the course page. :contentReference[oaicite:0]{index=0}


## Team Members

List of team members are here:

- Arafat Rahman
- Md Sabbir Ahmed


## Learning Goals

This topic focuses on understanding when to use RAG, what each component of a RAG pipeline does, and how retrieval parameters (chunking, top-k, prompt templates) affect quality and latency. :contentReference[oaicite:1]{index=1}


## Setup (Exercise 0)

1) Download and run the provided notebook:
- manual_rag_pipeline_universal.ipynb (linked on the course page)

2) Download and unzip the corpora:
- Corpora.zip (linked on the course page)

Notes from course updates (important for getting good retrieval results):
- Re-download Corpora.zip and use the “NewModelT” files (cleaner text) rather than older low-quality OCR. Prefer the .txt versions. :contentReference[oaicite:2]{index=2}
- The Congressional Record corpus can have garbled tables after extraction; some questions that depend on tables may not work well without improved preprocessing. :contentReference[oaicite:3]{index=3}
- In general, use the provided .txt files rather than extracting embedded PDF text in-code if possible. :contentReference[oaicite:4]{index=4}


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

We compared answers from a small open model:
- without retrieval (no RAG),
- with retrieval augmentation (RAG pipeline),

using two corpora separately:
- Model T repair manual
- Congressional Record corpus

The course page provides example queries for both corpora. :contentReference[oaicite:5]{index=5}


### Exercise 2 — Open Model + RAG vs Large Model (No RAG)

We wrote/adapted a small script to query GPT-4o Mini (no tools, single-turn questions) and compared its answers to the Exercise 1 queries.

We documented:
- hallucination avoidance relative to the small open model,
- which questions GPT-4o Mini got right without retrieval,
- how this relates to the corpora’s time period vs the model’s training cut-off. :contentReference[oaicite:6]{index=6}


### Exercise 3 — Local RAG vs Frontier Chat Model

We compared:
- Local: small open model + RAG (Model T corpus)
- Cloud: a frontier chat model via web interface (no file upload)

We documented:
- where the frontier model’s general knowledge succeeds,
- where local RAG provides more specific/grounded answers,
- cases where the frontier model seems to use live web search. :contentReference[oaicite:7]{index=7}


### Exercise 4 — Effect of Top-K Retrieval Count

We varied the number of retrieved chunks:
k = 1, 3, 5, 10, 20

For each k, we ran the same 3–5 queries and recorded:
- answer quality (accuracy, completeness),
- latency,
- when additional context stopped helping or started hurting. :contentReference[oaicite:8]{index=8}


### Exercise 5 — Handling Unanswerable Questions

We tested unanswerable questions such as:
- completely off-topic,
- related but not in corpus,
- false-premise questions.

We documented:
- whether the model admits it cannot answer,
- whether it hallucinates,
- whether irrelevant retrieved context makes hallucination more likely.

We also tested prompt-template changes such as:
“If the context doesn’t contain the answer, say ‘I cannot answer this from the available documents.’” :contentReference[oaicite:9]{index=9}


### Exercise 6 — Query Phrasing Sensitivity

We selected one underlying question and rephrased it 5+ ways (formal, casual, keywords-only, etc.), then recorded:
- top retrieved chunks,
- similarity scores,
- overlap between retrieval sets,
- which phrasing retrieved best context.

This experiment is used to reason about query rewriting strategies. :contentReference[oaicite:10]{index=10}


## Repository Organization

Suggested structure (adapt if your folder already differs):

- notebooks/
  - manual_rag_pipeline_universal.ipynb (or your modified copy)
- corpora/
  - (unzipped Corpora.zip contents, preferably using provided .txt files)
- outputs/
  - exercise1/
  - exercise2/
  - exercise3/
  - exercise4/
  - exercise5/
  - exercise6/
- notes/
  - findings.md (short write-up across experiments)
  - pitfalls.md (preprocessing issues, table failures, etc.)

We ensured each team member has a copy of results in their GitHub, per the assignment instructions. :contentReference[oaicite:11]{index=11}


## How To Run

Option A (recommended by course): run on Colab, especially for long-running re-indexing experiments. :contentReference[oaicite:12]{index=12}

Option B (local): run the notebook locally.

Typical workflow:
1) Open the notebook.
2) Point the ingestion step to the desired .txt corpus.
3) Build the index (embedding + chunking).
4) Run queries with and without retrieval augmentation.
5) Save outputs into outputs/ and summarize observations in notes/.


## Notes

Preprocessing quality is decisive for retrieval quality, especially with multi-column PDFs and tables. The course page explicitly notes this limitation and recommends using the provided text corpora (and the cleaner Model T text). :contentReference[oaicite:13]{index=13}
