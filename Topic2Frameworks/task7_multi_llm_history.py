# task7_multi_llm_history.py
# Task 7: Multi-LLM chat history with Human, Llama, Qwen.
# Three entities mapped to user/assistant roles via content prefixes:
#   Human: "Human: ..." (user role)
#   Llama: "Llama: ..." (assistant when Llama replied; user when context for Qwen)
#   Qwen:  "Qwen: ..."  (assistant when Qwen replied; user when context for Llama)
# Routing: "Hey Qwen" -> Qwen; else -> Llama.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

SYSTEM_LLAMA = (
    "You are Llama. You are in a conversation with a human and Qwen (another AI). "
    "When the human says 'Hey Qwen', they are addressing Qwen. "
    "Always start your responses with 'Llama: ' followed by your answer."
)
SYSTEM_QWEN = (
    "You are Qwen. You are in a conversation with a human and Llama (another AI). "
    "When the human says 'Hey Qwen', they are addressing you. "
    "Always start your responses with 'Qwen: ' followed by your answer."
)

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
    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str
    should_exit: bool
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
        return_full_text=False,
    )
    print("Llama model loaded successfully!")
    return HuggingFacePipeline(pipeline=pipe)

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
        return_full_text=False,
    )
    print("Qwen model loaded successfully!")
    return HuggingFacePipeline(pipeline=pipe)

def _format_for_llama(messages: list[BaseMessage]) -> list[dict]:
    """Build chat format for Llama: Human=user, Llama=assistant, Qwen=user."""
    out = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            out.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            out.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            c = str(msg.content)
            if c.strip().lower().startswith("llama:"):
                out.append({"role": "assistant", "content": c})
            else:
                out.append({"role": "user", "content": c})
    return out

def _format_for_qwen(messages: list[BaseMessage]) -> list[dict]:
    """Build chat format for Qwen: Human=user, Llama=user, Qwen=assistant."""
    out = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            out.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            out.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            c = str(msg.content)
            if c.strip().lower().startswith("qwen:"):
                out.append({"role": "assistant", "content": c})
            else:
                out.append({"role": "user", "content": c})  # Llama's reply as user for Qwen
    return out

def _clean_response(text: str, prompt_with_gen=None) -> str:
    s = str(text).strip()
    if prompt_with_gen and s.startswith(prompt_with_gen):
        s = s[len(prompt_with_gen):].strip()
    for tok in ["<|start_header_id|>assistant<|end_header_id|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]:
        s = s.replace(tok, "").strip()
    return s

def create_graph(llama_llm, qwen_llm):
    def get_user_input(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        print("\n" + "=" * 50)
        print("Enter text (or 'quit', 'verbose', 'quiet'). Say 'Hey Qwen' to address Qwen:")
        print("=" * 50)
        print("\n> ", end="")
        user_input = input()
        is_empty = not user_input.strip()
        if is_empty:
            print("Please enter some text (empty input ignored).")
            return {"user_input": user_input, "should_exit": False, "verbose": verbose, "skip_llm": False, "is_empty_input": True}
        if user_input.lower() in ["quit", "exit", "q"]:
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
        human_msg = HumanMessage(content=f"Human: {user_input}")
        return {"messages": [human_msg], "user_input": user_input, "should_exit": False, "verbose": verbose, "skip_llm": False, "is_empty_input": False}

    def call_llama(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        messages = state.get("messages", [])
        if not messages:
            messages = []
        msgs_for_llama = [{"role": "system", "content": SYSTEM_LLAMA}] + _format_for_llama(messages)
        tokenizer = llama_llm.pipeline.tokenizer
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(msgs_for_llama, tokenize=False, add_generation_prompt=True)
            prompt_with_gen = prompt
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in msgs_for_llama) + "\nassistant: "
            prompt_with_gen = None
        print("\nProcessing with Llama...")
        response = llama_llm.invoke(prompt)
        content = response.content if isinstance(response, BaseMessage) else str(response)
        cleaned = _clean_response(content, prompt_with_gen)
        if not cleaned.lower().startswith("llama:"):
            cleaned = f"Llama: {cleaned}"
        return {"messages": [AIMessage(content=cleaned)]}

    def call_qwen(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        messages = state.get("messages", [])
        if not messages:
            messages = []
        msgs_for_qwen = [{"role": "system", "content": SYSTEM_QWEN}] + _format_for_qwen(messages)
        tokenizer = qwen_llm.pipeline.tokenizer
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(msgs_for_qwen, tokenize=False, add_generation_prompt=True)
            prompt_with_gen = prompt
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in msgs_for_qwen) + "\nassistant: "
            prompt_with_gen = None
        print("\nProcessing with Qwen...")
        response = qwen_llm.invoke(prompt)
        content = response.content if isinstance(response, BaseMessage) else str(response)
        cleaned = _clean_response(content, prompt_with_gen)
        if not cleaned.lower().startswith("qwen:"):
            cleaned = f"Qwen: {cleaned}"
        return {"messages": [AIMessage(content=cleaned)]}

    def print_response(state: AgentState) -> dict:
        skip_llm = state.get("skip_llm", False)
        if skip_llm:
            return {}
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                print("\n" + "=" * 50)
                print("RESPONSE")
                print("=" * 50)
                print(msg.content)
                print("=" * 50)
                break
        return {}

    def route_after_input(state: AgentState):
        if state.get("is_empty_input", False):
            return "get_user_input"  # Loop back, never pass empty to LLM
        if state.get("should_exit", False):
            return END
        if state.get("skip_llm", False):
            return "print_response"
        if state.get("user_input", "").strip().lower().startswith("hey qwen"):
            return "call_qwen"
        return "call_llama"

    builder = StateGraph(AgentState)
    builder.add_node("get_user_input", get_user_input)
    builder.add_node("call_llama", call_llama)
    builder.add_node("call_qwen", call_qwen)
    builder.add_node("print_response", print_response)
    builder.add_edge(START, "get_user_input")
    builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {"get_user_input": "get_user_input", "call_llama": "call_llama", "call_qwen": "call_qwen", "print_response": "print_response", END: END},
    )
    builder.add_edge("call_llama", "print_response")
    builder.add_edge("call_qwen", "print_response")
    builder.add_edge("print_response", "get_user_input")
    return builder.compile()

def main():
    print("=" * 50)
    print("Multi-LLM Chat (Llama + Qwen, shared history)")
    print("=" * 50)
    print("\nLoading models...")
    llama_llm = create_llama_llm()
    qwen_llm = create_qwen_llm()
    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm, qwen_llm)
    print("Graph created successfully!")
    initial_state: AgentState = {
        "messages": [],
        "user_input": "",
        "should_exit": False,
        "verbose": False,
        "skip_llm": False,
        "is_empty_input": False,
    }
    graph.invoke(initial_state)

if __name__ == "__main__":
    main()
