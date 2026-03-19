# Topic2Frameworks

Solutions for Agent Control Flow Frameworks (Agentic AI Spring 2026, UVA).  
[Course page](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic2Frameworks/Agent%20Control%20Flow%20Frameworks.html)

## Table of Contents

| File | Description |
|------|-------------|
| `README.md` | This file |
| `requirements.txt` | Python dependencies |
| `task2_verbose_quiet.py` | Task 2: verbose/quiet tracing |
| `task3_empty_input.py` | Task 3: empty input handling |
| `task4_parallel_models.py` | Task 4: parallel Llama + Qwen |
| `task5_selective_routing.py` | Task 5: "Hey Qwen" routing |
| `task6_chat_history.py` | Task 6: chat history (Llama only) |
| `task7_multi_llm_history.py` | Task 7: multi-LLM chat |
| `task8_checkpoint_recovery.py` | Task 8: checkpointing + crash recovery |
| `task3_empty_output.txt` | Recorded empty-input behavior |
| `task4_output.txt` | Sample output from Task 4 |
| `task7_output.txt` | Sample output from Task 7 |
| `task7_sample_conversation.txt` | Sample multi-LLM conversation |
| `task8_output.txt` | Sample output from Task 8 |
| `lg_graph.mmd` | Graph visualization (Mermaid) |
| `tests/` | Unit tests |
| `tests/test_routing.py` | Routing logic tests |
| `tests/test_task7_helpers.py` | Task 7 helper tests |
| `tests/conftest.py` | Pytest configuration |

## Setup

```bash
python -m pip install -r requirements.txt
```

## Tests

```bash
pytest tests/ -v
```

## Tasks

| Task | File | Description |
|------|------|-------------|
| 2 | `task2_verbose_quiet.py` | "verbose" / "quiet" tracing toggle |
| 3 | `task3_empty_input.py` | Empty input → 3-way branch, loop back; never pass empty to LLM |
| 4 | `task4_parallel_models.py` | pass_input → Llama + Qwen in parallel; print both results |
| 5 | `task5_selective_routing.py` | "Hey Qwen" → Qwen; else → Llama |
| 6 | `task6_chat_history.py` | Chat history (Message API: system, human, ai; Llama only) |
| 7 | `task7_multi_llm_history.py` | Multi-LLM chat: Human/Llama/Qwen prefixes; "Hey Qwen" → Qwen |
| 8 | `task8_checkpoint_recovery.py` | Checkpointing + crash recovery; SqliteSaver; resume from last node |

`task3_empty_output.txt` — Recorded empty-input behavior for Task 3.

## Run

```bash
python task2_verbose_quiet.py
python task3_empty_input.py
python task4_parallel_models.py
python task5_selective_routing.py
python task6_chat_history.py
python task7_multi_llm_history.py
python task8_checkpoint_recovery.py
```

**Task 8** (with output saved; use `LD_LIBRARY_PATH` for conda envs with libstdc++ ABI issues):
```bash
# Interactive: see output and save to file
LD_LIBRARY_PATH=$CONDA_PREFIX/lib python task8_checkpoint_recovery.py 2>&1 | tee task8_output.txt

# Non-interactive: save output to file only
LD_LIBRARY_PATH=$CONDA_PREFIX/lib python task8_checkpoint_recovery.py > task8_output.txt 2>&1
```

## Examples

**Task 4** (parallel — both models run):
```bash
echo -e "What is 2+2?\nquit" | python task4_parallel_models.py
```

**Task 5** (selective — one model per input):
```bash
echo -e "What is 2+2?\nHey Qwen, what is 3+3?\nquit" | python task5_selective_routing.py
```
First query → Llama; second → Qwen.

**Task 6** (chat history — Message API, Llama only):
```bash
echo -e "My name is Alice\nWhat is my name?\nquit" | python task6_chat_history.py
```
Maintains context; second query uses prior messages.

**Task 7** (multi-LLM chat — Human, Llama, Qwen with shared history):
```bash
echo -e "What is the best ice cream flavor?\nHey Qwen, what do you think?\nI agree.\nquit" | python task7_multi_llm_history.py
```
First → Llama; second → Qwen (with Llama's reply in context); third → Llama (with Qwen's reply in context).

**Task 8** (checkpoint recovery — kill mid-conversation, restart to resume):
```bash
LD_LIBRARY_PATH=$CONDA_PREFIX/lib python task8_checkpoint_recovery.py 2>&1 | tee task8_output.txt
```
Start a conversation, type a message, then Ctrl+C. Run again — it resumes from the last checkpoint.
