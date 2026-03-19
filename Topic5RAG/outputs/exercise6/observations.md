# Exercise 6 Observations

## Top-5 retrieved chunks by phrasing

See phrasing_results.md for full output.

## Similarity scores (top chunk)

| Phrasing | Top score | Notes |
|----------|-----------|-------|
| formal | 0.5976 | Oil change 400/750 miles, cold weather 500 miles — relevant |
| casual | 0.6859 | "Every ten days" service, oil change — best score, relevant |
| keywords | 0.5357 | Oil change 500 miles — relevant |
| question_form | 0.5742 | Magnets, starting crank — wrong! "Check" → diagnostic, not schedule |
| indirect | 0.4302 | Generic "essentials of good service" — weakest, less relevant |

## Overlap between result sets

- **formal ∩ casual: 3/5** — strong overlap; both retrieved maintenance schedule content
- **casual ∩ keywords: 3/5** — strong overlap
- **formal ∩ keywords: 2/5**
- **question_form, indirect:** 0 overlap with each other and with formal/casual/keywords — retrieved completely different content

## Which phrasing retrieved the best chunks?

**Casual** ("How often should I service the engine?") — highest score (0.6859), retrieved oil change intervals and "every ten days" service. **Formal** and **keywords** also retrieved relevant maintenance schedule content. **Question_form** semantically confused "check" with diagnostic procedure (magnets). **Indirect** retrieved generic service quality, not specific intervals.

## Do keyword-style queries work better or worse than natural questions?

**Natural (casual) worked best.** Keywords (0.5357) retrieved relevant content but scored lower than casual (0.6859). Formal (0.5976) was comparable to keywords. So natural questions can outperform keywords when phrased conversationally.

## Implications for query rewriting strategies

- Different phrasings can yield **zero overlap** — "check the engine" vs "maintenance schedule" retrieved entirely different chunks.
- Query rewriting could normalize ambiguous phrases: "when do I need to check" → "maintenance schedule" or "service intervals" to avoid semantic drift.
- Expanding vague terms ("preventive maintenance requirements") to specific concepts ("oil change intervals", "service schedule") may improve retrieval.
- Casual phrasing matched corpus language ("every ten days", "change oil") — aligning query style with document style helps.

## Summary

- **Best phrasing:** Casual. Natural questions can outperform keywords.
- **Semantic confusion:** "Check" retrieved diagnostic procedure, not maintenance schedule — phrasing matters.
- **Query rewriting:** Normalize ambiguous terms; expand vague to specific; align with document style.
