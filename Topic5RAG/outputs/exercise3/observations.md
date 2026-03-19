# Exercise 3 Observations

## Comparison: Local RAG (Qwen + Model T) vs Frontier Model (GPT-5.3, no RAG)

| Query | Local RAG better? | Frontier better? | Frontier used web search? |
|-------|------------------|------------------|---------------------------|
| Carburetor adjustment | Yes (exact manual: rod, cotter pin, throttle lever B) | Partially (comprehensive context) | No |
| Spark plug gap | No (OCR error: 7/4") | Yes (0.025" correct) | No |
| Slipping transmission band | Yes (manual-specific: Fig. 132, 299) | Partially (broader context) | No |
| Engine oil | No (retrieval failed) | Yes (SAE 30 non-detergent) | No |

## Where frontier model's general knowledge succeeds

- **Spark plug gap:** 0.025" with magneto context, feeler gauge procedure — correct. Local RAG had OCR/parsing error.
- **Engine oil:** SAE 30 non-detergent, splash lubrication, no filter, dippers — correct vintage Model T knowledge. Local RAG retrieval failed (Learjet docs).
- **Carburetor, transmission:** Frontier gave comprehensive Model T–specific context (Kingston/Holley, planetary bands, steering levers) even without the manual.

## Where local RAG provides more accurate, specific answers

- **Carburetor:** Exact manual procedure — adjusting rod, forked end, needle valve, cotter pin, slot in dash, throttle lever "B". Frontier gave general procedure; RAG grounded in manual text.
- **Transmission band:** Manual-specific — slow-speed adjusting screw lock nut, Fig. 132, Fig. 299. Frontier gave broader context (low/reverse/brake bands); RAG cited exact figures.

## Cases where frontier model appeared to use live web search

No apparent web search. No citations or "I searched" language. Answers appear to rely on training knowledge. Model T is historical (within any cutoff).

## Summary

- **When RAG adds value:** Manual-specific procedures (carburetor rod, transmission Fig. 132) — RAG grounds answers in the exact source text.
- **When a powerful model suffices:** General vintage-car knowledge (spark plug gap, engine oil) — frontier model has strong training knowledge; RAG can fail (OCR, wrong corpus).
- **RAG failures hurt:** Engine oil retrieval failed; frontier model rescued the answer.
