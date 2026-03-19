# task4_parallel_models.py
# Task 4: Output from get_user_input goes to pass_input node, which passes input to BOTH
# Llama and Qwen. The models run in parallel. print_both_responses prints both results.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

def get_device():
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    else:
        print("Using CPU for inference")
        return "cpu"

class AgentState(TypedDict):
    user_input: str
    should_exit: bool
    llama_response: str
    qwen_response: str
    verbose: bool
    skip_llm: bool
    is_empty_input: bool

def create_llama_llm():
    device = get_device()
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"Loading Llama model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )
    if device == "mps":
        model = model.to(device)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    print("Llama model loaded successfully!")
    return llm

def create_qwen_llm():
    device = get_device()
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading Qwen model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )
    if device == "mps":
        model = model.to(device)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    print("Qwen model loaded successfully!")
    return llm

def create_graph(llama_llm, qwen_llm):
    def get_user_input(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        if verbose:
            print("\n[TRACE] Entering node: get_user_input")
        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit, 'verbose' for tracing, 'quiet' to disable tracing):")
        print("=" * 50)
        print("\n> ", end="")
        user_input = input()
        is_empty = not user_input.strip()
        if is_empty:
            if verbose:
                print("[TRACE] Empty input detected - will loop back to get_user_input")
            print("Please enter some text (empty input ignored).")
            return {"user_input": user_input, "should_exit": False, "verbose": verbose, "skip_llm": False, "is_empty_input": True}
        if user_input.lower() in ['quit', 'exit', 'q']:
            if verbose:
                print("[TRACE] User requested exit")
            print("Goodbye!")
            return {"user_input": user_input, "should_exit": True, "verbose": verbose, "skip_llm": False, "is_empty_input": False}
        if user_input.lower() == "verbose":
            if not verbose:
                print("Verbose tracing enabled")
            return {"user_input": user_input, "should_exit": False, "verbose": True, "skip_llm": True, "is_empty_input": False}
        if user_input.lower() == "quiet":
            if verbose:
                print("[TRACE] Disabling verbose mode")
            else:
                print("Verbose tracing disabled")
            return {"user_input": user_input, "should_exit": False, "verbose": False, "skip_llm": True, "is_empty_input": False}
        if verbose:
            print(f"[TRACE] User input received: '{user_input}'")
            print("[TRACE] Routing to: pass_input (will fan out to Llama and Qwen in parallel)")
        return {"user_input": user_input, "should_exit": False, "verbose": verbose, "skip_llm": False, "is_empty_input": False}

    def pass_input(state: AgentState) -> dict:
        """Pass input to both Llama and Qwen. Both models run in parallel."""
        verbose = state.get("verbose", False)
        if verbose:
            print("\n[TRACE] Entering node: pass_input")
            print("[TRACE] Passing input to both Llama and Qwen (parallel execution)")
        return {}

    def call_llama(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        user_input = state["user_input"]
        if verbose:
            print("\n[TRACE] Entering node: call_llama")
        prompt = f"User: {user_input}\nAssistant:"
        print("\nProcessing with Llama...")
        response = llama_llm.invoke(prompt)
        if verbose:
            print("[TRACE] Llama response received")
        return {"llama_response": response}

    def call_qwen(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        user_input = state["user_input"]
        if verbose:
            print("\n[TRACE] Entering node: call_qwen")
        prompt = f"User: {user_input}\nAssistant:"
        print("\nProcessing with Qwen...")
        response = qwen_llm.invoke(prompt)
        if verbose:
            print("[TRACE] Qwen response received")
        return {"qwen_response": response}

    def print_both_responses(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        if verbose:
            print("\n[TRACE] Entering node: print_both_responses")
        llama_response = state.get("llama_response", "")
        qwen_response = state.get("qwen_response", "")
        print("\n" + "=" * 50)
        print("MODEL RESPONSES")
        print("=" * 50)
        print("\n" + "-" * 50)
        print("Llama Response:")
        print("-" * 50)
        print(llama_response or "(none)")
        print("\n" + "-" * 50)
        print("Qwen Response:")
        print("-" * 50)
        print(qwen_response or "(none)")
        print("\n" + "=" * 50)
        if verbose:
            print("[TRACE] Both responses printed")
        return {}

    def print_response(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        skip_llm = state.get("skip_llm", False)
        if verbose:
            print("\n[TRACE] Entering node: print_response")
        if skip_llm:
            if verbose:
                print("[TRACE] LLMs were skipped (verbose/quiet command)")
            return {}
        return {}

    def route_after_input(state: AgentState):
        if state.get("is_empty_input", False):
            return "get_user_input"
        if state.get("should_exit", False):
            return END
        if state.get("skip_llm", False):
            return "print_response"
        return "pass_input"

    def route_from_pass_input(state: AgentState):
        """Fan out to both Llama and Qwen - they run in parallel."""
        return ["call_llama", "call_qwen"]

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("pass_input", pass_input)
    graph_builder.add_node("call_llama", call_llama)
    graph_builder.add_node("call_qwen", call_qwen)
    graph_builder.add_node("print_both_responses", print_both_responses)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "get_user_input": "get_user_input",
            "pass_input": "pass_input",
            "print_response": "print_response",
            END: END,
        }
    )
    graph_builder.add_conditional_edges("pass_input", route_from_pass_input)
    graph_builder.add_edge("call_llama", "print_both_responses")
    graph_builder.add_edge("call_qwen", "print_both_responses")
    graph_builder.add_edge("print_both_responses", "get_user_input")
    graph_builder.add_edge("print_response", "get_user_input")

    return graph_builder.compile()

def main():
    print("=" * 50)
    print("LangGraph Agent with Llama and Qwen (parallel)")
    print("=" * 50)
    print()
    print("Loading models...")
    llama_llm = create_llama_llm()
    qwen_llm = create_qwen_llm()
    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm, qwen_llm)
    print("Graph created successfully!")
    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "llama_response": "",
        "qwen_response": "",
        "verbose": False,
        "skip_llm": False,
        "is_empty_input": False,
    }
    graph.invoke(initial_state)

if __name__ == "__main__":
    main()
