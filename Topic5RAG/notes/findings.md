# RAG Experiments — Findings

Short write-up across experiments. Update after each exercise.

---

## Exercise 1 — RAG vs No-RAG

We compared Qwen 2.5 1.5B with and without RAG on the Model T Ford manual and Congressional Record (Jan 2026).

### Model T corpus
- **Carburetor, transmission band:** No-RAG gave generic modern-car advice; RAG produced manual-specific instructions (adjusting rod, needle valve, lock nut, etc.).
- **Spark plug gap:** No-RAG hallucinated 0.25"; RAG cited manual but had OCR/parsing error (7/4" vs likely 7/64").
- **Engine oil:** No-RAG said 10W-30 (wrong for Model T). RAG retrieved Learjet docs and gave a nonsensical answer — retrieval failure (wrong corpus or mixed index).

### Congressional Record corpus
- **Mr. Flood / Mayor Black (Jan 13):** No-RAG invented that the mayor was "not present"; RAG correctly described Flood’s recognition of Black and Papillion.
- **Elise Stefanik (Jan 23):** No-RAG fabricated Trump/Obama/Bush misattribution; RAG correctly stated the context lacked the answer.
- **Main Street Parity Act (Jan 20):** No-RAG cited a 2013 bill; RAG correctly described the Jan 2026 bill (SBA, 504 loans, 10% equity).
- **Pregnancy centers (Jan 21):** No-RAG mixed real names with wrong context; RAG correctly summarized Republicans for and Democrats against.

### Takeaways
- RAG strongly reduced hallucinations when retrieval succeeded.
- Jan 2026 CR is beyond training cutoff, so general knowledge alone fails.
- Retrieval can fail (wrong corpus, OCR) and produce bad answers.
- When retrieval finds nothing relevant, the model can correctly decline to answer.

---

## Exercise 2 — Open Model + RAG vs Large Model (GPT-4o Mini)

We ran GPT-4o Mini (no RAG) on the same 8 queries and compared to Qwen 2.5 1.5B (no RAG).

### Model T corpus
- **Spark plug gap:** GPT-4o Mini said 0.025" (plausible); Qwen said 0.25" (wrong). GPT-4o Mini better.
- **Engine oil:** GPT-4o Mini said SAE 30/40 non-detergent (correct for vintage); Qwen said 10W-30 only. GPT-4o Mini better.
- **Carburetor, transmission:** Both gave generic advice; neither manual-specific. GPT-4o Mini slightly more plausible (hand crank, mixture screw).

### Congressional Record (Jan 2026)
- **Mr. Flood / Mayor Black, Elise Stefanik:** GPT-4o Mini **correctly declined** ("no access after October 2023"). Qwen fabricated. GPT-4o Mini much better at avoiding hallucination.
- **Main Street Parity Act:** Both wrong. GPT-4o Mini described a different bill (community banks, regulatory burden); the Jan 2026 bill is about SBA 504 loans, 10% equity.
- **Pregnancy centers:** GPT-4o Mini gave correct general pattern (Republicans for, Democrats against) and acknowledged cutoff. Qwen mixed real names with wrong context.

### Takeaways
- GPT-4o Mini avoids hallucination by **declining** when beyond its cutoff; Qwen fabricated.
- For Model T (within training), GPT-4o Mini has better vintage-car knowledge.
- For post-cutoff events, neither knows; GPT-4o Mini's honesty is preferable to Qwen's fabrication.

---

## Exercise 3 — Local RAG vs Frontier Chat Model

We compared local Qwen+RAG (Model T) vs GPT-5.3 (no RAG) via web interface.

### Where frontier model's general knowledge succeeds
- **Spark plug gap:** 0.025" with magneto context — correct. Local RAG had OCR error (7/4").
- **Engine oil:** SAE 30 non-detergent, splash lubrication — correct. Local RAG retrieval failed (Learjet docs).
- **Carburetor, transmission:** Frontier gave comprehensive Model T context (needle valves, planetary bands) without the manual.

### Where local RAG provides more accurate, specific answers
- **Carburetor:** Exact manual procedure — adjusting rod, cotter pin, throttle lever "B". RAG grounded in source.
- **Transmission band:** Manual-specific — Fig. 132, Fig. 299, slow-speed screw. RAG cited exact figures.

### Frontier model and web search
No apparent live web search. Answers relied on training. Model T is historical.

### Takeaways
- RAG adds value for manual-specific procedures (exact steps, figure references).
- Frontier model can outperform when RAG fails (retrieval error, wrong corpus) or when general vintage knowledge suffices.

---

## Exercise 4 — Effect of Top-K

We varied k = 1, 3, 5, 10, 20 on 4 Model T queries.

### Answer quality
- **Carburetor, transmission:** More context helped (k=5–20). Best answers at k=10–20 with full procedure and figure refs.
- **Spark plug:** k=1, k=3 honest ("not specified"). k=5 gave wrong .006"–.010" (coil gap). k=10, k=20 hallucinated "2 inches."
- **Engine oil:** k=1 honest. k≥3 wrong — retrieved "gallon of new oil" drain instruction; model conflated quantity with oil type.

### Latency
- k=1: ~4.7s avg (fastest). k=3: ~14s (high variance). k=5–20: ~6.5–8.6s. No strict linear increase with k.

### Takeaways
- More context can hurt when retrieval returns irrelevant chunks; the LLM may hallucinate from them.
- For noisy retrieval, lower k (e.g., k=1) can yield honest "I don't know" instead of wrong answers.
- Optimal k depends on retrieval quality. k=5–10 is a reasonable default for well-retrieved queries.

---

## Exercise 5 — Unanswerable Questions

We tested 3 unanswerable questions with default and strict prompts ("If the context doesn't contain the answer, say 'I cannot answer this from the available documents.'").

### Does the model admit it doesn't know?
- **Off-topic (capital of France):** Yes — both prompts. Strict produced clean "I cannot answer."
- **Related (1925 hp):** No — both hallucinated "~20 hp" (plausible but not in corpus).
- **False premise (synthetic oil):** No — both invented reasons the manual recommends synthetic oil (it doesn't).

### Does irrelevant context encourage hallucination?
Yes. For related and false-premise questions, retrieved Model T chunks (oil, cylinders, rings) gave the model something to "explain" from, leading to plausible-sounding but wrong answers.

### Does the strict prompt help?
Only for off-topic. For related and false-premise, the model ignored "do not use general knowledge" and hallucinated.

### Takeaways
- Strict prompt helps when the question is clearly outside the corpus. For topical-but-unanswerable questions, it often fails.
- Irrelevant but related context can encourage hallucination by providing material to misinterpret.

---

## Exercise 6 — Query Phrasing Sensitivity

We tested 5 phrasings of "engine maintenance" on the Model T corpus.

### Similarity scores and relevance
- **Casual** ("How often should I service the engine?"): 0.686 — best score; retrieved oil change intervals, "every ten days" service.
- **Formal, keywords:** Also retrieved relevant maintenance content (400/750 miles, 500 cold weather).
- **Question_form** ("When do I need to check the engine?"): Retrieved magnet-checking, diagnostic procedure — semantic confusion ("check" → inspect, not schedule).
- **Indirect** ("Preventive maintenance requirements"): Weakest (0.43); generic service quality, not specific intervals.

### Overlap
- formal ∩ casual: 3/5; casual ∩ keywords: 3/5 — strong overlap for maintenance-focused phrasings.
- question_form, indirect: 0 overlap with others — retrieved entirely different content.

### Takeaways
- Natural casual phrasing outperformed keywords. Phrasing significantly affects retrieval.
- Ambiguous terms ("check") can cause semantic drift; query rewriting to normalize (e.g., "maintenance schedule") may help.
- Aligning query style with document language improves retrieval.

---

## Exercise 7 — Chunk Overlap

We tested overlap = 0, 64, 128, 256 (chunk_size=512) on a boundary-spanning query: oil drain procedure.

### Index size and retrieval

- **n_chunks:** 0 → 2320; 64 → 2716; 128 → 3277; 256 → 5215 (+125% at 256 vs 0)
- **Top scores:** 0.624 (0), 0.648 (64), 0.635 (128), 0.664 (256)
- **Answer completeness:** Overlap 0 retrieved core "drain and pour" but missed pipe-cleaning step (Fig. 294). Overlap 256 retrieved the full procedure: blow out oil pipe, drain, pour gallon through breather. Overlap 128 gave a good balance.

### Takeaways

- Higher overlap improves retrieval when answers span chunk boundaries (multi-step procedures).
- Index size grows sharply (2.25× at overlap 256); overlap 128 is a practical default.
- Trade-off: completeness vs. index size and redundancy.

---

## Exercise 8 — Chunk Size

We tested chunk_size = 128, 256, 512, 1024, 2048 (overlap scales) on the oil drain procedure query.

### Index size and retrieval

- **n_chunks:** 128 → 19254; 256 → 10305; 512 → 3277; 1024 → 1333; 2048 → 645 (~30× fewer at 2048 vs 128)
- **Top scores:** 0.774 (128), 0.700 (256), 0.635 (512), 0.603 (1024), 0.583 (2048) — smaller chunks had higher similarity
- **Answer completeness:** 512 gave best balance — full procedure (Fig. 294, drain, pour, time gear). 128 had best scores but missed pipe-cleaning step (fragmented). 2048 mixed in tangential content (radiator, hood).

### Takeaways

- Smaller chunks improve retrieval scores (more focused matches); larger chunks dilute similarity.
- 512 is a practical default for technical manuals — complete procedures without excess noise.
- Trade-off: granularity (precision) vs. context per chunk (completeness).

---

## Exercise 9 — Retrieval Score Analysis

We ran 10 queries, retrieved top-10 chunks each, and analyzed score distribution.

### Score distribution

- **Overall:** Min 0.31 | Max 0.67 | Mean 0.47 | Std 0.08
- **Highest top scores:** Engine knocking (0.67), magneto (0.65), carburetor (0.64)
- **Lowest top scores:** Engine oil (0.46), maintenance schedule (0.46)
- **Smallest spread (Std 0.02):** Engine overhaul — consistent relevance across top-10
- **Largest spread (Std 0.14):** Engine knocking — steep drop from top 2 to tail; engine oil had sharp drop #1→#2 (retrieval difficulty)

### Takeaways

- Top score < 0.5 may indicate weak retrieval; > 0.6 with small std = confident retrieval.
- Score distribution useful for thresholds and diagnosing retrieval issues.

---

## Exercise 10 — Prompt Template Variations

We tested 5 prompt templates (minimal, strict, citation, permissive, structured) on 3 queries.

### Template comparison

- **Most accurate:** Strict and citation for carburetor (manual-specific procedure). All failed spark plug (OCR). Engine oil: strict/citation stayed grounded; permissive hallucinated "diesel fuel".
- **Best at avoiding hallucination:** Strict — when retrieval failed (engine oil), stayed with context; permissive invented wrong info.
- **Citation:** Improved traceability (Chapter XXVII, 878; Chapter X, 493).
- **Structured:** Improved usability for carburetor (numbered steps).

### Takeaways

- Strict grounding reduces hallucination when retrieval is weak; permissive is risky.
- Citation improves traceability; structured improves usability for procedures.

---

## Exercise 11 — Cross-Document Synthesis

We tested synthesis queries (maintenance tasks, compare procedures, tools for tune-up, safety warnings) with top_k = 3, 5, 10.

### Synthesis quality

- **Combined info:** Yes for carburetor vs transmission comparison and tools — coherent synthesis from multiple chunks.
- **Higher top_k:** Mixed — tools: k=5 helped, k=10 added redundancy; safety: k=5 hallucinated warnings, k=10 correctly said "none."
- **Missed info:** "Monthly maintenance" retrieved parts/inventory content, not vehicle maintenance (oil change, "every ten days").
- **Contradictions:** Safety warnings — k=5 listed 8 warnings; k=10 said "no explicit safety warnings." Same query, different top_k → opposite answers.

### Takeaways

- Model can synthesize when retrieval is good; retrieval mismatch ("monthly") yields wrong focus.
- Higher k does not always help — can add noise or change answer.
- "ALL" / "summarize all" queries risk missing info if retrieval fails; model may not acknowledge gaps.

