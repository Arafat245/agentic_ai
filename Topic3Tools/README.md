# Topic3Tools — Agent Tool Use

This directory contains my solutions for Topic 3: Agent Tool Use from the Agentic AI Spring 2026 course (CS 6501, University of Virginia).

The goal of this topic is to connect an LLM-driven agent to external tools (both local and API-based), then integrate tool calling into a LangGraph control-flow program.


## Overview

In this topic, we:

- Set up a local LLM server (Ollama) and measure runtime behavior.
- Verify OpenAI API-based model access using environment variables.
- Implement a custom calculator tool (JSON in, JSON out).
- Add multiple tools and dispatch them cleanly (tool map instead of long if/else).
- Run a tool-enabled LangGraph agent.
- Extend the graph into a persistent conversation (stateful chat loop).
- Reflect on parallelization opportunities in a tool-using agent graph.

Assignment reference:
https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic3Tools/tools.html


## Requirements

Install dependencies:

pip install -r requirements.txt

You will also need:

- Ollama installed and running (for local model tasks).
- OPENAI_API_KEY exported (for OpenAI tasks).


## Environment Setup

OpenAI API key (for the GPT-4o-mini task):

export OPENAI_API_KEY="your_key_here"

If you are using Ollama (local LLM), ensure Ollama is installed and your model is pulled (example):

ollama pull llama3.2:1b

Then run Ollama server in a separate terminal (if needed by your setup):

ollama serve


## Files

This folder contains code for the required tasks. Each task is implemented as a separate script for clarity and grading.

Starter / provided / referenced scripts (if present in the repo):
- requirements.txt

Your task scripts (recommended naming):

Task 1:
- task1_ollama_timing.py

Task 2:
- task2_openai_gpt4o_mini_test.py

Task 3:
- task3_manual_calculator_tool.py

Task 4:
- task4_langgraph_tools_basic.py

Task 5:
- task5_multi_tool_queries.py

Task 6:
- task6_langgraph_persistent_conversation.py

Task 7:
- task7_parallelization_discussion.txt


## Tasks and What Each Script Does

Task 1 — Ollama Setup and Timing
File: task1_ollama_timing.py

- Starts from the provided benchmark scripts approach.
- Measures runtime of two runs sequentially and then in parallel.
- Uses shell timing (time) or Python timing.
- Records observed speed differences and notes why parallel may or may not help.


Task 2 — OpenAI GPT-4o-mini Test
File: task2_openai_gpt4o_mini_test.py

- Verifies OPENAI_API_KEY is available.
- Calls GPT-4o-mini and prints a short response.
- Confirms your OpenAI client environment is correct.


Task 3 — Manual Calculator Tool (JSON)
File: task3_manual_calculator_tool.py

- Implements a calculator tool that accepts a JSON string as input.
- Uses json.loads to parse arguments.
- Returns output using json.dumps.
- Includes extra operations beyond +, -, *, / (as required by the assignment prompt).


Task 4 — LangGraph Agent with Multiple Tools
File: task4_langgraph_tools_basic.py

- Implements 3 tools:
  - Calculator tool
  - Letter-count tool
  - One additional custom tool
- Uses a tool map for dispatch.
- Integrates tool calling into a LangGraph agent loop.


Task 5 — Multi-tool Queries
File: task5_multi_tool_queries.py

- Demonstrates prompts that cause the agent to invoke more than one tool.
- For example:
  - compute something and then count letters
  - chained tool usage within a single turn


Task 6 — Persistent Conversation via LangGraph
File: task6_langgraph_persistent_conversation.py

- Rewrites the program as a stateful LangGraph conversation:
  - maintains conversation state (history/messages)
  - loops until the user exits
  - tool calls happen inside the graph
- Adds checkpointing/recovery if required by the assignment prompt.


Task 7 — Parallelization Discussion
File: task7_parallelization_discussion.txt

- Written response explaining which parts of your tool-enabled agent could run in parallel.
- Notes practical constraints:
  - tool dependencies
  - LLM latency
  - shared state


## Running

From inside this directory, run:

python task1_ollama_timing.py
python task2_openai_gpt4o_mini_test.py
python task3_manual_calculator_tool.py
python task4_langgraph_tools_basic.py
python task5_multi_tool_queries.py
python task6_langgraph_persistent_conversation.py


## Expected Behavior

- Task 1 prints sequential vs parallel runtime measurements.
- Task 2 prints a valid short response from GPT-4o-mini.
- Task 3 prints correct JSON outputs from calculator requests.
- Task 4 runs an interactive agent that can call tools.
- Task 5 demonstrates multi-tool invocations.
- Task 6 runs a persistent conversation loop with tool calling in the graph.
- Task 7 is a short write-up (no code).
