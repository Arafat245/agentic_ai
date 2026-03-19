# Exercise 11 Observations

## Synthesis results by top_k

See synthesis_results.md for full output.

## Can the model successfully combine information from multiple chunks?

Yes for well-retrieved queries. **Carburetor vs transmission comparison:** k=3, 5, 10 all produced coherent comparisons combining steps from both procedures. **Tools for tune-up:** k=5 and k=10 synthesized tools from multiple divisions (cylinder, engine, transmission, axle). **Maintenance tasks:** Retrieved parts/inventory content (steel bins, 90-day turnover) instead of vehicle maintenance (oil change, "every ten days") — retrieval mismatch for "monthly".

## Does retrieving more chunks improve synthesis?

Mixed. **Tools:** k=5 added useful tools; k=10 added redundancy (repeated "crankcase reamer tool"). **Comparison:** k=5 and k=10 gave more detailed procedures than k=3. **Safety warnings:** k=5 hallucinated 8 specific warnings; k=10 correctly said "no explicit safety warnings" — more chunks sometimes reduced hallucination by providing broader context.

## Does it miss information that wasn't retrieved?

Yes. **Maintenance tasks:** Query "monthly" retrieved dealer/parts-inventory chunks, not oil-change intervals or "every ten days" service. Model answered from what was retrieved; it did not acknowledge that vehicle maintenance (from Ex 6) was missing. **Safety warnings:** k=3 and k=10 acknowledged "no explicit safety warnings"; k=5 did not — it invented them.

## Does contradictory information in different chunks cause problems?

Yes. **Safety warnings:** k=5 listed 8 specific warnings (moving parts, gloves, etc.); k=10 said "no safety warnings." Same query, different top_k → opposite answers. k=5 likely retrieved chunks that prompted the model to invent plausible-sounding warnings; k=10's broader context led to correct "none" answer. Contradictions can arise from retrieval variance.

## Implications for synthesis queries

- Higher top_k does not always improve synthesis — can add noise or redundancy.
- Query phrasing matters: "monthly maintenance" retrieved different content than "maintenance schedule."
- For "ALL" or "summarize all" queries, model may not acknowledge gaps when retrieval misses relevant chunks.
- Synthesis quality depends on retrieval; consider multi-query or query expansion for broad synthesis.

## Summary

- Model can combine info when retrieval is good (comparison, tools).
- Higher k: sometimes helps, sometimes adds noise, sometimes changes answer (safety).
- Missed retrieval → model answers from what it has; may not acknowledge gaps.
- Contradictions possible across top_k due to retrieval variance.
