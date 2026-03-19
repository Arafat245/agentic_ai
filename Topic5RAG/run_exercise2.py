#!/usr/bin/env python3
"""
Exercise 2: Open Model + RAG vs Large Model (GPT-4o Mini)

Queries GPT-4o Mini with NO retrieval on the same 8 queries from Exercise 1.
Compare to Qwen 2.5 1.5B (no RAG) to assess hallucination avoidance.

Requires: pip install openai
          OPENAI_API_KEY environment variable

Run from: Topic5RAG/ directory
    python run_exercise2.py
"""

import os
from pathlib import Path

# Same queries as Exercise 1
QUERIES_MODEL_T = [
    "How do I adjust the carburetor on a Model T?",
    "What is the correct spark plug gap for a Model T Ford?",
    "How do I fix a slipping transmission band?",
    "What oil should I use in a Model T engine?",
]
QUERIES_CR = [
    "What did Mr. Flood have to say about Mayor David Black in Congress on January 13, 2026?",
    "What mistake did Elise Stefanik make in Congress on January 23, 2026?",
    "What is the purpose of the Main Street Parity Act?",
    "Who in Congress has spoken for and against funding of pregnancy centers?",
]

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "exercise2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def query_gpt4o_mini(question: str, model: str = "gpt-4o-mini") -> str:
    """Single-turn question to GPT-4o Mini, no tools, no RAG."""
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        max_tokens=512,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable.")
        print("  export OPENAI_API_KEY='your-key-here'")
        return 1

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: Install openai package: pip install openai")
        return 1

    lines = [
        "# Exercise 2: GPT-4o Mini (No RAG)",
        "# Same queries as Exercise 1 — compare to Qwen 2.5 1.5B no-RAG answers",
        "",
    ]

    # Model T queries
    lines.append("## Model T Queries")
    lines.append("")
    for i, q in enumerate(QUERIES_MODEL_T, 1):
        lines.append(f"### Query {i}: {q}")
        lines.append("")
        print(f"Querying GPT-4o Mini: {q[:50]}...")
        try:
            ans = query_gpt4o_mini(q)
            lines.append(ans)
        except Exception as e:
            lines.append(f"[Error: {e}]")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Congressional Record queries
    lines.append("## Congressional Record Queries")
    lines.append("")
    for i, q in enumerate(QUERIES_CR, 1):
        lines.append(f"### Query {i}: {q}")
        lines.append("")
        print(f"Querying GPT-4o Mini: {q[:50]}...")
        try:
            ans = query_gpt4o_mini(q)
            lines.append(ans)
        except Exception as e:
            lines.append(f"[Error: {e}]")
        lines.append("")
        lines.append("---")
        lines.append("")

    outpath = OUTPUT_DIR / "gpt4o_mini_results.md"
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n✓ Saved to {outpath}")

    # Create observations template
    obs = """# Exercise 2 Observations

## Comparison: GPT-4o Mini (no RAG) vs Qwen 2.5 1.5B (no RAG)

| Query | Qwen hallucinated? | GPT-4o Mini correct? | Notes |
|-------|--------------------|----------------------|-------|
| Carburetor adjustment | | | |
| Spark plug gap | | | |
| Slipping transmission band | | | |
| Engine oil | | | |
| Mr. Flood / Mayor Black (Jan 13) | | | |
| Elise Stefanik (Jan 23) | | | |
| Main Street Parity Act (Jan 20) | | | |
| Pregnancy centers (Jan 21) | | | |

## Training Cutoff vs Corpus Age

- GPT-4o Mini training cutoff: [check model card]
- Model T manual: historical (1908–1927)
- Congressional Record: January 2026 (after any LLM cutoff)

## Summary

- Which questions does GPT-4o Mini answer correctly without RAG?
- Does GPT-4o Mini avoid hallucinations better than Qwen 2.5 1.5B?
- For Jan 2026 CR: can GPT-4o Mini know these events? (web search vs training?)
"""
    obs_path = OUTPUT_DIR / "observations.md"
    with open(obs_path, "w", encoding="utf-8") as f:
        f.write(obs)
    print(f"✓ Created {obs_path} — fill in after comparing to Exercise 1 results.")

    return 0


if __name__ == "__main__":
    exit(main())
