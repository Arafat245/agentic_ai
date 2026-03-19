# Exercise 1 — How to Run

## Prerequisites

1. Open `manual_rag_pipeline_universal.ipynb` in Jupyter or Colab.
2. Run all cells **through** the RAG pipeline (cells 1–32), so that:
   - Documents are loaded from `Corpora/NewModelT`
   - Chunking, embedding, and FAISS index are built
   - Qwen 2.5 1.5B and `direct_query` / `rag_query` are available

## Run Order

Execute these cells in order:

| Cell | Description |
|------|-------------|
| **Exercise 1 Part A** | Runs 4 Model T queries (no RAG + RAG), saves `model_t_results.txt` |
| **Switch to CR** | Loads Congressional Record, re-chunks, re-embeds, rebuilds index |
| **Exercise 1 Part B** | Runs 4 Congressional Record queries, saves `congressional_record_results.txt` |
| **Exercise 1 Part C** | Creates `observations.md` template |

## Output Files

After running, you will have:

- `model_t_results.txt` — Model T queries, no-RAG and RAG answers, retrieved chunks
- `congressional_record_results.txt` — Congressional Record queries, same format
- `observations.md` — Table template to fill in (hallucination, grounding, general knowledge)

## Notes

- Part A uses the **current** index. If you loaded NewModelT earlier, it will run Model T queries.
- Part B switches to Congressional Record and rebuilds the index (takes a few minutes).
- Fill in `observations.md` after reviewing the results.
- Update `notes/findings.md` with your summary.
