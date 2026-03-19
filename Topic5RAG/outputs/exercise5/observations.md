# Exercise 5 Observations

## Does the model admit it doesn't know?

| Question type | Default prompt | Strict prompt |
|---------------|----------------|---------------|
| Off-topic (capital of France) | Yes (said no mention, cannot determine) | Yes ("I cannot answer this from the available documents.") |
| Related not in corpus (1925 hp) | No (said "~20 hp" from context) | No (same — claimed 20 hp, inferred) |
| False premise (synthetic oil) | No (invented reasons for synthetic oil) | No (same — invented viscosity benefits) |

## Does it hallucinate plausible-sounding but wrong answers?

- **Off-topic:** No — correctly declined. Default cited irrelevant "xiv Service follow-up" but did not answer.
- **Related (1925 hp):** Yes — "~20 hp" is plausible (Model T was ~20 hp) but not in corpus. Model claimed it was from context.
- **False premise:** Yes — Manual does NOT recommend synthetic oil. Model invented "synthetic oil because... viscosity, 500-mile changes" and even added a signature. Severe hallucination.

## Does retrieved context help or hurt?

- **Off-topic:** Retrieved irrelevant chunks (document structure). Model correctly did not use them to answer.
- **Related, false premise:** Retrieved Model T–related chunks (cylinders, oil, rings). Model used them to justify wrong answers — irrelevant/partial context encouraged hallucination by giving something to "explain" from.

## Does the strict prompt help?

- **Off-topic:** Yes — strict produced clean "I cannot answer this from the available documents."
- **Related, false premise:** No — strict did not change behavior. Model still used general knowledge and hallucinated despite "Do NOT use your general knowledge."

## Summary

- Strict prompt helps only when the question is clearly off-topic. For related or false-premise questions, the model still hallucinates.
- Irrelevant but topical context (oil, cylinders) encourages hallucination — model finds something to latch onto.
- Small models may ignore "do not use general knowledge" when the question feels answerable from training.
