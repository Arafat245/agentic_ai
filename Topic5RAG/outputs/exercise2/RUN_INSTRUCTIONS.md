# Exercise 2 — How to Run

## Prerequisites

1. **OpenAI API key** — Get one from https://platform.openai.com/api-keys
2. **openai package** — `pip install openai`
3. **Exercise 1 results** — For comparison (Qwen no-RAG answers in `outputs/exercise1/`)

## Where to Run

**Option A — Terminal (recommended):**

```bash
cd /path/to/agentic_ai/Topic5RAG
export OPENAI_API_KEY='your-key-here'
python run_exercise2.py
```

**Option B — Notebook:**

1. Open `manual_rag_pipeline_universal.ipynb`
2. Run cells 1–9 (to define `QUERIES_MODEL_T`, `QUERIES_CR`)
3. In a terminal: `export OPENAI_API_KEY='your-key-here'` (or set in notebook)
4. Run the **Exercise 2** cell (after Exercise 1 cells)

## What It Does

- Queries GPT-4o Mini (no tools, no RAG) on the same 8 queries from Exercise 1
- Saves answers to `gpt4o_mini_results.md`
- Creates `observations.md` template for comparison

## Output Files

- `gpt4o_mini_results.md` — All 8 queries and GPT-4o Mini answers
- `observations.md` — Template to fill in (compare to Qwen, training cutoff notes)

## After Running

1. Compare `gpt4o_mini_results.md` to `outputs/exercise1/model_t_results.txt` and `congressional_record_results.txt` (no-RAG sections)
2. Fill in `observations.md` — which questions did GPT-4o Mini get right? Better hallucination avoidance?
3. Update `notes/findings.md` with Exercise 2 summary
