# RAG Experiments — Pitfalls

Preprocessing issues, table failures, and other lessons learned.

---

## Preprocessing

- **Congressional Record:** Multi-column text is single-column in the txt/ version, but tables can be garbled when extracting embedded text. RAG may fail for questions that depend on table content.
- **Model T manual:** Use NewModelT (cleaner text) rather than older OCR versions.
- **Learjet:** Embedded text is medium quality; text on diagrams is not useful.

---

## Corpus-Specific Notes

- **Model T engine oil:** When DOC_FOLDER pointed to the parent `Corpora/` folder instead of `Corpora/NewModelT`, the index included Learjet and other corpora. The "engine oil" query then retrieved ATA_12 (Learjet) instead of Model T, producing a nonsensical answer. Always point DOC_FOLDER to the specific corpus folder (e.g., `Corpora/NewModelT`) for single-corpus experiments.
