# Exercise 2 Observations

## Comparison: GPT-4o Mini (no RAG) vs Qwen 2.5 1.5B (no RAG)

| Query | Qwen hallucinated? | GPT-4o Mini correct? | Notes |
|-------|--------------------|----------------------|-------|
| Carburetor adjustment | Yes (generic modern-car) | Partially (generic but more plausible) | Both lack manual specifics; GPT-4o Mini mentions hand crank, mixture screw |
| Spark plug gap | Yes (0.25" — wrong) | Yes (0.025" — plausible) | GPT-4o Mini much better; 0.025" is in range for Model T |
| Slipping transmission band | Yes (generic) | Partially (generic) | Both gave generic advice; neither manual-specific |
| Engine oil | Yes (10W-30 only) | Yes (SAE 30/40 non-detergent) | GPT-4o Mini correct for vintage Model T |
| Mr. Flood / Mayor Black (Jan 13) | Yes (invented "mayor not present") | Correctly declined | GPT-4o Mini: "no access after Oct 2023" — honest |
| Elise Stefanik (Jan 23) | Yes (fabricated Trump/Obama/Bush) | Correctly declined | GPT-4o Mini: cutoff; Qwen fabricated |
| Main Street Parity Act (Jan 20) | Yes (2013 bill) | No (different bill) | GPT-4o Mini described community-bank bill; Jan 2026 bill is SBA/504 loans |
| Pregnancy centers (Jan 21) | Yes (wrong context) | Partially (general pattern + cutoff) | GPT-4o Mini: Republicans for, Democrats against; acknowledged Oct 2023 cutoff |

## Training Cutoff vs Corpus Age

- GPT-4o Mini training cutoff: October 2023 (stated in its responses)
- Model T manual: historical (1908–1927) — within training
- Congressional Record: January 2026 — after cutoff

## Summary

- **GPT-4o Mini answers correctly without RAG:** Spark plug gap, engine oil (Model T). For Jan 2026 CR, it correctly declines (Flood, Stefanik) instead of fabricating.
- **GPT-4o Mini avoids hallucinations better:** Yes. It declines when beyond cutoff; Qwen fabricated specific false details. For Model T, GPT-4o Mini has better vintage-car knowledge (0.025" gap, non-detergent SAE 30/40).
- **Jan 2026 CR:** GPT-4o Mini does not use web search; it states its cutoff and declines. Main Street Parity Act: it described a different (older) bill from training. Pregnancy centers: general pattern correct, acknowledged cutoff.
