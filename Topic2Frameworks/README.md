Topic2Frameworks — Agent Control Flow Frameworks

This directory contains my solutions for Topic 2: Agent Control Flow Frameworks from the Agentic AI Spring 2026 course (CS 6501).
The goal of this topic is to explore the LangGraph control flow abstraction for orchestrating autonomous agent workflows using LLMs.

Table of Contents

Introduction

Requirements

Starter Agent

Tasks and Scripts

Running Instructions

Expected Behavior

Notes

Introduction

Agent control flow frameworks specify how an agent routes state between discrete processing steps.
In this topic, we use LangGraph, a graph-based framework built on LangChain, to define agent workflows with nodes, conditional edges, parallel execution, and history tracking. The starter code langgraph_simple_llama_agent.py demonstrates a looping agent that reads user input, invokes a Hugging Face model, and prints responses.

Requirements

The following packages are required to run the code in this directory:

torch
transformers
langchain-huggingface
langgraph
grandalf

Install them using:

pip install -r requirements.txt
Starter Agent

The file langgraph_simple_llama_agent.py demonstrates:

A basic LangGraph with nodes: get_user_input, call_llm, and print_response.

A router that continuously loops until the user requests exit.

A Hugging Face LLM (meta-llama/Llama-3.2-1B-Instruct) to generate responses.

Tracing toggles (verbose, quiet).

Use this file as the foundation for subsequent tasks.

Tasks and Scripts

Below are the required tasks with suggested filenames and a brief description of the implemented behavior.

Task	Script	Description
1	task1_verbose_quiet.py	Extend starter agent so that input verbose turns node tracing on and quiet turns it off.
2	task2_handle_empty_input.py	Ensure empty user input is not passed to the LLM by modifying the get_user_input node and routing.
3	task3_three_way_branch.py	Add a three-way branch from get_user_input including an edge back to itself.
4	task4_parallel_models.py	Extend graph to run both Llama and a second model (e.g., Qwen) in parallel and print both results.
5	task5_selective_routing.py	Modify routing so that input starting with “Hey Qwen” is sent to Qwen, otherwise to Llama.
6	task6_history_tracking.py	Add chat history tracking using the Message API. Disable Qwen for this task.
7	task7_multi_llm_history.py	Integrate chat history for both Llama and Qwen with proper role handling and system prompts.

Each script should contain your modified agent implementation and comment at the top describing changes made to satisfy the task requirements. Task numbering follows the assignment page tasks.

Running Instructions

From this directory, run scripts as:

python task<N>_<description>.py

For example:

python task1_verbose_quiet.py

Ensure your Python environment has the required packages installed.

Expected Behavior

For the starter agent (langgraph_simple_llama_agent.py):

Prompt appears for user input.

Typing normal text routes input through the graph to the LLM and prints the response.

Typing verbose toggles on detailed trace output from nodes.

Typing quiet turns trace output off.

Typing quit, exit, or q exits the agent.

For task scripts:

Task 1 — Commands verbose and quiet control tracing.

Task 2 — Empty input should be recognized and not forwarded to the LLM.

Task 3 — The graph should branch back to prompt for new input when appropriate.

Task 4 — Two model calls should run in parallel, and both outputs should be printed.

Task 5 — Routing logic should send relevant inputs to the selected model.

Task 6 — Chat history should accumulate and be passed to the model.

Task 7 — History with multiple agents should be integrated with appropriate roles.

Notes

LangGraph supports checkpoints and crash recovery; this is part of the optional Task 8 and is not covered here.

