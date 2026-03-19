# Exercise 1 Observations

## Model T Queries
| Query | Hallucinates (no RAG)? | RAG grounds answer? | General knowledge correct? |
|-------|------------------------|---------------------|----------------------------|
| Carburetor adjustment | Yes (generic modern-car steps) | Yes (manual-specific rod, needle valve, cotter pin) | Partially (generic concepts only) |
| Spark plug gap | Yes (0.25" — wrong) | Partially (7/4" likely OCR for 7/64"; dime thickness correct) | No |
| Slipping transmission band | Yes (generic transmission advice) | Yes (slow-speed adjusting screw, lock nut from manual) | Partially (general concepts) |
| Engine oil | Yes (10W-30 — wrong for Model T) | No (retrieved Learjet manual; wrong corpus) | Partially (10W-30 exists but not for Model T) |

## Congressional Record Queries
| Query | Hallucinates (no RAG)? | RAG grounds answer? | General knowledge correct? |
|-------|------------------------|---------------------|----------------------------|
| Mr. Flood / Mayor Black (Jan 13) | Yes (said mayor "not present") | Yes (Flood recognized Black, 17½ years, Papillion) | No (Jan 2026 after cutoff) |
| Elise Stefanik mistake (Jan 23) | Yes (fabricated Trump/Obama/Bush) | Yes (correctly said context lacks details) | No |
| Main Street Parity Act (Jan 20) | Yes (2013 bill that died) | Yes (SBA, 504 loans, 10% equity) | No (different bill) |
| Pregnancy centers funding (Jan 21) | Yes (mixed real names, wrong context) | Yes (Republicans for, Democrats against) | Partially (real politicians, wrong facts) |

## Summary
- **RAG helped most:** Carburetor, transmission band, Mr. Flood/Black, Main Street Parity Act, pregnancy centers — RAG grounded answers in corpus and corrected hallucinations.
- **RAG failed:** Engine oil — retrieved Learjet docs instead of Model T (corpus mix or path issue). Spark plug gap — RAG had OCR/parsing error (7/4" vs 7/64").
- **RAG appropriately declined:** Elise Stefanik — no relevant chunks for Jan 23; model correctly said it cannot answer.
- **General knowledge never sufficient** for Congressional Record (Jan 2026 is after training cutoff). For Model T, generic knowledge was often wrong (e.g., 10W-30, 0.25" gap).
