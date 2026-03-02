# Topic2Frameworks — Agent Control Flow Frameworks

This directory contains my solutions for Topic 2: Agent Control Flow Frameworks from the Agentic AI Spring 2026 course (CS 6501, University of Virginia).

The objective of this topic is to explore how agent control flow can be modeled explicitly using LangGraph. We progressively extend a simple LLM-based agent into a multi-model, conditional, stateful conversational graph.

The starter reference comes from the course page:
https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic2Frameworks/Agent%20Control%20Flow%20Frameworks.html


## Overview

In this topic, we:

- Build an agent using LangGraph.
- Route state between nodes explicitly.
- Add conditional branching.
- Run multiple models in parallel.
- Implement selective routing.
- Maintain chat history across turns.
- Integrate multi-model conversational state.

All tasks are implemented as separate scripts for clarity.


## Requirements

Install dependencies using:

pip install -r requirements.txt

The requirements file contains:

torch
transformers
langchain-huggingface
langgraph
grandalf


## Starter Agent

The file:

langgraph_simple_llama_agent.py

Provides:

- A minimal LangGraph agent.
- Nodes:
  - get_user_input
  - call_llm
  - print_response
- A routing function controlling flow.
- A HuggingFace LLM backend (Llama).
- Verbose tracing toggle.

This file serves as the base for all tasks.


## Tasks and Implementations

Task 1 — Verbose / Quiet Toggle  
File: task1_verbose_quiet.py  

Adds control commands:
- "verbose" enables tracing.
- "quiet" disables tracing.

The graph execution reflects trace visibility dynamically.


Task 2 — Handle Empty Input  
File: task2_handle_empty_input.py  

Modifies routing logic so that:
- Empty user input is not forwarded to the LLM.
- Control returns to input node.


Task 3 — Three-Way Branch  
File: task3_three_way_branch.py  

Extends the graph to include:
- A three-way conditional branch.
- A path that routes back to get_user_input.
- Proper control-flow logic.


Task 4 — Parallel Models  
File: task4_parallel_models.py  

Extends the graph to:
- Run Llama and Qwen in parallel.
- Print both model responses.
- Maintain synchronized state.


Task 5 — Selective Routing  
File: task5_selective_routing.py  

Implements routing logic:

- If input starts with "Hey Qwen" → route to Qwen.
- Otherwise → route to Llama.

Ensures only the selected model runs.


Task 6 — Chat History (Single Model)  
File: task6_history_tracking.py  

Adds:
- Chat message history.
- LangChain Message API usage.
- Persistent conversation state.
- Qwen disabled for this step.


Task 7 — Multi-LLM Chat History  
File: task7_multi_llm_history.py  

Extends history support to:

- Both Llama and Qwen.
- Proper role-based formatting.
- Multi-agent conversational memory.
- System prompt handling.


## Running

From inside this directory:

python task1_verbose_quiet.py
python task2_handle_empty_input.py
python task3_three_way_branch.py
python task4_parallel_models.py
python task5_selective_routing.py
python task6_history_tracking.py
python task7_multi_llm_history.py


## Expected Behavior

Starter Agent:

- Prompts user for input.
- Sends text to LLM.
- Prints response.
- "verbose" shows node-level tracing.
- "quiet" hides tracing.
- "quit", "exit", or "q" exits loop.

Task Extensions:

- Task 1: Trace toggle works dynamically.
- Task 2: Empty input is ignored.
- Task 3: Graph supports explicit three-way branching.
- Task 4: Two models run in parallel.
- Task 5: Routing selects model conditionally.
- Task 6: Chat history persists across turns.
- Task 7: Multi-model conversation memory maintained.


## Notes

- Each task is implemented as a separate script for grading clarity.
- Graph visualization is optional but supported via grandalf.
- Optional checkpointing task is not included here.

This directory demonstrates progressively more sophisticated agent orchestration patterns using LangGraph.
