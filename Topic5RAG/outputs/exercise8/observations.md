# Exercise 8 Observations

## Top-5 retrieved chunks by chunk size

See chunk_size_results.md for full output.

## Index size (n_chunks) by chunk size

| Chunk size | Overlap | n_chunks | Notes |
|------------|---------|----------|-------|
| 128 | 64 | 19254 | Most granular |
| 256 | 128 | 10305 | |
| 512 | 128 | 3277 | |
| 1024 | 128 | 1333 | |
| 2048 | 128 | 645 | ~30× fewer than 128 |

## Similarity scores (top chunk)

| Chunk size | Top score | Notes |
|------------|-----------|-------|
| 128 | 0.7744 | Highest — focused chunks match query well |
| 256 | 0.7001 | |
| 512 | 0.6347 | |
| 1024 | 0.6025 | |
| 2048 | 0.5825 | Lowest — larger chunks dilute similarity |

## Does larger chunk size improve retrieval of complete information?

Not linearly. **Smaller chunks** (128, 256) had higher similarity scores — more focused, precise matches. **Larger chunks** (512, 1024, 2048) had lower scores but each chunk contained more context. For the oil-drain procedure: 512 retrieved Fig. 294 (pipe cleaning), drain, pour, time gear, cylinder cover — good completeness. 128 had best scores but the answer missed the pipe-cleaning step (procedure split across chunks). 2048 retrieved chapter-level content ("Cleaning the Oil Line") but the answer mixed in tangential steps (radiator, hood). **512 gave the best balance** — complete procedure without excess noise.

## Cost: Index size, context granularity

Index size drops sharply with larger chunks: 19254 → 645 (~30×). Fewer chunks = faster embedding and retrieval. Trade-off: smaller chunks = higher precision but fragmented context; larger chunks = more context per chunk but diluted similarity and potential noise.

## Point of diminishing returns?

**512 is a good plateau.** Smaller (128, 256) improve scores but fragment procedures; larger (1024, 2048) reduce index size but add noise and lower retrieval precision. 512 captured the full oil-drain procedure (Fig. 294, drain, pour, time gear) with moderate index size (3277 chunks).

## Implications for chunking strategy

- Smaller chunks improve retrieval *scores* but can split procedures across boundaries — overlap helps (Ex 7).
- 512 is a practical default for technical manuals: enough context per chunk, reasonable index size.
- Very large chunks (2048) risk chapter-level retrieval with mixed, tangential content.

## Summary

- Smaller chunks → higher similarity scores; larger chunks → lower scores (dilution).
- 512 balanced completeness and index size; 128 fragmented the procedure.
- Trade-off: granularity (precision) vs. context per chunk (completeness).
