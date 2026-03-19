# Exercise 7 Observations

## Top-5 retrieved chunks by overlap

See overlap_results.md for full output.

## Index size (n_chunks) by overlap

| Overlap | n_chunks | Notes |
|---------|----------|-------|
| 0 | 2320 | Baseline, no redundancy |
| 64 | 2716 | +17% vs 0 |
| 128 | 3277 | +41% vs 0 |
| 256 | 5215 | +125% vs 0 |

## Similarity scores (top chunk)

| Overlap | Top score | Notes |
|---------|-----------|-------|
| 0 | 0.6237 | Direct drain/pour chunk |
| 64 | 0.6479 | Oil pipe cleaning (Fig. 294) — different procedure |
| 128 | 0.6347 | Drain + pour in top chunks |
| 256 | 0.6638 | Best score; full procedure (blow pipe, drain, pour) |

## Does higher overlap improve retrieval of complete information?

Yes. The oil-drain procedure spans boundaries: (1) blow out oil pipe with compressed air (Fig. 294), (2) drain old oil, (3) pour gallon of new oil through breather. Overlap 0 retrieved the core "drain and pour" but missed the pipe-cleaning step. Overlap 256 retrieved chunks that span the full sequence — top chunk includes "blow out any foreign matter... drain... pour." Overlap 128 gave a clearer answer structure than 64; overlap 256 gave the most complete procedure with air-hose and drain steps.

## Cost: Index size, redundant information

Index size grows sharply: 2320 → 5215 chunks (2.25× at overlap 256). Higher overlap creates more chunks with redundant content — the same sentences appear in multiple overlapping chunks. This increases embedding cost, storage, and retrieval noise (similar chunks may rank multiple times).

## Point of diminishing returns?

Overlap 128 vs 256: top score improved (0.6347 → 0.6638) and answer completeness improved, but index nearly doubled (3277 → 5215). Overlap 64 had the second-best top score (0.6479) but retrieved oil-pipe cleaning rather than the core drain procedure. **Reasonable plateau: overlap 128** — good answer quality without the full index blow-up of 256. For critical boundary-spanning procedures, 256 may be worth the cost.

## Implications for chunking strategy

- For boundary-spanning answers (multi-step procedures), overlap helps — zero overlap can split key steps across chunks.
- Overlap 64–128 is a practical default; 256 when completeness matters and index size is acceptable.
- Trade-off: more overlap → more complete retrieval vs. larger index and redundancy.

## Summary

- Higher overlap improved retrieval of the full oil-drain procedure (pipe cleaning + drain + pour).
- Index size grew 2.25× from overlap 0 to 256; diminishing returns after ~128.
- Overlap 128 is a reasonable default; 256 for critical procedures when completeness matters.
