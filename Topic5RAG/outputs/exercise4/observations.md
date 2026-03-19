# Exercise 4 Observations

## Answer Quality by top_k

| k | Carburetor | Spark plug | Transmission | Engine oil | Notes |
|---|------------|------------|--------------|------------|-------|
| 1 | Good (rod, cotter) | Honest (not specified) | OK (Fig 132, 299) | Honest (no info) | Minimal context; honest when missing |
| 3 | Good | Honest | Good | Wrong (gallon) | Irrelevant chunk confused oil |
| 5 | Good | Wrong (.006–.010") | Good | Wrong (gallon) | Spark: confused coil gap |
| 10 | Best (4 steps) | Wrong (2") | Good (Fig 311, 315) | Wrong (gallon) | Spark: severe hallucination |
| 20 | Best (5 steps) | Wrong (2") | Best (most figures) | Wrong (gallon) | More context → more errors for bad retrieval |

## Latency

| k | Avg (s) | Range |
|---|---------|-------|
| 1 | 4.69 | 3.49–6.10 |
| 3 | 13.97 | 5.01–24.15 (high variance) |
| 5 | 6.56 | 3.26–9.71 |
| 10 | 7.24 | 2.71–10.51 |
| 20 | 8.63 | 6.53–10.79 |

Latency does not scale linearly with k. k=3 had outliers (21s, 24s). k=1 was fastest.

## When does adding context stop helping?

- **Carburetor, transmission:** Quality improved up to k=10–20. More context helped.
- **Spark plug, engine oil:** Quality peaked at k=1–3. Higher k introduced wrong values.

## When does too much context hurt?

- **Spark plug:** k=5 gave .006"–.010" (coil/magnet gap, not spark plug). k=10, k=20 gave "2 inches" — clear hallucination from misread table.
- **Engine oil:** k≥3 retrieved "pour in a gallon of new oil" (drain instruction). Model inferred "gallon of new oil" as the answer instead of oil type. k=1 correctly said context lacked the info.

## Summary

- **Optimal k:** Depends on retrieval quality. For good retrieval (carburetor, transmission): k=5–10. For noisy retrieval: k=1 can yield honest "I don't know."
- **Trade-off:** More context can introduce irrelevant chunks; the LLM may hallucinate from them. Latency is variable, not strictly proportional to k.
