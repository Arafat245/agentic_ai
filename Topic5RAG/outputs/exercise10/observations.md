# Exercise 10 Observations

## Answers by template

See prompt_results.md for full output.

## Which template produced the most accurate answers?

**Carburetor:** Strict and citation gave the most manual-specific procedures (adjusting rod, priming wires, throttle lever). Minimal mixed in unrelated steps (horn, engine pan). **Spark plug:** All failed — corpus has OCR errors; minimal gave 0.006–0.010" (likely coil gap); others hallucinated "2 inches" from misreading "Ys". **Engine oil:** Corpus lacks oil type (retrieval failure). Strict and citation stayed grounded ("gallon of new oil" from context); minimal and permissive hallucinated (Shell DTE-100, diesel fuel).

## Which template best avoided hallucination?

**Strict grounding** best avoided hallucination when retrieval failed. For engine oil, strict said "gallon of new oil" (from context, wrong but grounded); **permissive** invented "No. 2 light diesel fuel" — completely wrong. For spark plug, all templates failed due to OCR; strict did not help. **Takeaway:** Strict reduces hallucination on weak-retrieval queries; permissive can add dangerous wrong info.

## Did citation improve traceability?

Yes for carburetor — citation template cited "Chapter XXVII, paragraph 878", "ModelTNew.pdf". For engine oil, citation cited "Chapter X, Section 493" — traceable even when answer was wrong. Citation improves auditability.

## Did structured format improve usability?

Yes for carburetor — structured produced clear numbered steps (Step 1–8) that were easier to follow. For spark plug and engine oil, structured did not help accuracy (same underlying retrieval/OCR issues).

## Implications for prompt design

- Use **strict grounding** when retrieval quality is uncertain — reduces hallucination.
- **Citation** improves traceability; useful for technical manuals.
- **Structured** improves usability for procedural answers.
- **Permissive** is risky — can add wrong "general knowledge" (e.g., diesel for Model T).

## Summary

- Strict best avoided hallucination; permissive worst (diesel fuel for Model T).
- Citation improved traceability (chapter/paragraph refs).
- Structured improved usability for procedures.
- Prompt choice matters when retrieval fails — strict keeps answers grounded.
