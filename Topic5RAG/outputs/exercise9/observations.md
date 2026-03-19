# Exercise 9 Observations

## Top-10 retrieved chunks and scores by query

See retrieval_scores.md for full output.

## Score distribution summary

| Query | Top score | Min | Max | Mean | Std |
|-------|-----------|-----|-----|------|-----|
| Engine knocking | 0.6738 | 0.3061 | 0.6738 | 0.4554 | 0.1388 |
| Magneto strength | 0.6468 | 0.4116 | 0.6468 | 0.5122 | 0.0902 |
| Carburetor adjustment | 0.6392 | 0.5271 | 0.6392 | 0.5623 | 0.0370 |
| Oil drain procedure | 0.5825 | 0.4327 | 0.5825 | 0.4902 | 0.0500 |
| Rear axle | 0.5547 | 0.4621 | 0.5547 | 0.4954 | 0.0296 |
| Spark plug gap | 0.5511 | 0.4644 | 0.5511 | 0.4990 | 0.0319 |
| Engine overhaul | 0.5486 | 0.4857 | 0.5486 | 0.5061 | 0.0195 |
| Transmission band | 0.5404 | 0.4088 | 0.5404 | 0.4512 | 0.0481 |
| Maintenance schedule | 0.4551 | 0.3346 | 0.4551 | 0.3827 | 0.0422 |
| Engine oil | 0.4612 | 0.3429 | 0.4612 | 0.3608 | 0.0359 |

Overall: Min 0.31 | Max 0.67 | Mean 0.47 | Std 0.08

## Which queries had the highest/lowest top scores?

**Highest:** Engine knocking (0.67), magneto strength (0.65), carburetor (0.64) — procedural/specific queries with clear corpus match. **Lowest:** Engine oil (0.46), maintenance schedule (0.46) — engine oil had retrieval issues in Ex 1 (wrong corpus); maintenance may be spread across chunks.

## Score spread (top vs bottom of top-10)

**Smallest spread (Std 0.02):** Engine overhaul — top-10 chunks all similarly relevant (0.49–0.55). **Largest spread (Std 0.14):** Engine knocking — steep drop from top 2 (0.67) to #3 (0.54), then to tail (0.31). **Engine oil:** Sharp drop from #1 (0.46) to #2 (0.36) — only one chunk strongly matched; retrieval difficulty.

## Implications for retrieval quality

- Top score < 0.5 may indicate weak retrieval (engine oil, maintenance).
- Top score > 0.6 with small std = confident retrieval (carburetor, magneto).
- Large spread = top chunks relevant but tail has noise — consider lower top_k or score threshold.
- Steep drop after #1 = possible retrieval failure; verify top chunk relevance.

## Summary

- Highest scores: procedural queries (knocking, magneto, carburetor); lowest: engine oil, maintenance.
- Engine overhaul had most consistent top-10; engine knocking had steepest drop-off.
- Score distribution useful for setting confidence thresholds and diagnosing retrieval issues.
