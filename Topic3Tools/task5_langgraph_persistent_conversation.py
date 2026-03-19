"""
Task 5: Persistent LangGraph Conversation with Checkpointing

Single long conversation (no fresh start each turn).
Uses LangGraph nodes and edges instead of a Python loop.
Includes checkpointing and recovery (MemorySaver or SqliteSaver).
Tools: calculator, count_letter, reverse_words (from Task 4).
"""

import asyncio
import json
import math
import os
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Checkpointing: MemorySaver (in-memory) or SqliteSaver (persistent recovery)
CHECKPOINTER = None
CHECKPOINTER_TYPE = "none"

try:
    from langgraph.checkpoint.memory import MemorySaver
    CHECKPOINTER = MemorySaver()
    CHECKPOINTER_TYPE = "memory"
except ImportError:
    pass

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False

# Use SqliteSaver for persistent recovery if available and requested
USE_SQLITE = os.environ.get("TASK5_SQLITE", "").lower() in ("1", "true", "yes")
if HAS_SQLITE and USE_SQLITE:
    CHECKPOINTER = None  # Will be created as context manager
    CHECKPOINTER_TYPE = "sqlite"

# ============================================
# TOOLS (same as Task 4)
# ============================================

@tool
def calculator(expression: str) -> str:
    """Evaluate math expression: sin, cos, sqrt, etc. Example: sin(pi/2)"""
    try:
        try:
            args = json.loads(expression)
            expr = args.get("expression", args.get("expr", expression))
        except json.JSONDecodeError:
            expr = expression
        safe_dict = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
            "pi": math.pi, "e": math.e, "pow": math.pow, "abs": abs,
        }
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def count_letter(text: str, letter: str) -> str:
    """Count occurrences of a letter in text (case-insensitive)."""
    if len(letter) != 1:
        return json.dumps({"error": "letter must be a single character"})
    count = text.lower().count(letter.lower())
    return json.dumps({"text": text, "letter": letter, "count": count})


@tool
def reverse_words(text: str) -> str:
    """Reverse each word in the text."""
    words = text.split()
    return json.dumps({"original": text, "result": " ".join(w[::-1] for w in words)})


tools = [calculator, count_letter, reverse_words]
tool_map = {t.name: t for t in tools}

# ============================================
# STATE
# ============================================

class ConversationState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    verbose: bool
    command: str  # "exit", "verbose", "quiet", or None


# ============================================
# NODES
# ============================================

def input_node(state: ConversationState) -> dict:
    """Get user input and add to conversation."""
    if state.get("verbose", False):
        print("\n[NODE] input_node")
    user_input = input("\nYou: ").strip()

    if user_input.lower() in ["quit", "exit"]:
        return {"command": "exit"}
    if user_input.lower() == "verbose":
        return {"command": "verbose", "verbose": True}
    if user_input.lower() == "quiet":
        return {"command": "quiet", "verbose": False}

    return {"command": None, "messages": [HumanMessage(content=user_input)]}


def call_model(state: ConversationState) -> dict:
    """Call LLM with tools."""
    if state.get("verbose", False):
        print("\n[NODE] call_model")
    messages = list(state["messages"])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [
            SystemMessage(content="You are a helpful assistant. Use tools when needed. For math use calculator, for letter counts use count_letter.")
        ] + messages

    llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
    response = llm.invoke(messages)
    return {"messages": [response]}


def tools_node(state: ConversationState) -> dict:
    """Execute tool calls."""
    if state.get("verbose", False):
        print("\n[NODE] tools_node")
    last_message = state["messages"][-1]
    tool_messages = []
    for tc in last_message.tool_calls:
        name, args = tc["name"], tc["args"]
        if name in tool_map:
            if name == "calculator":
                result = tool_map[name].invoke(args.get("expression", json.dumps(args)))
            else:
                result = tool_map[name].invoke(args)
        else:
            result = f"Error: Unknown function {name}"
        if state.get("verbose", False):
            print(f"  Tool: {name}, Args: {args} -> {result[:80]}...")
        tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    return {"messages": tool_messages}


def output_node(state: ConversationState) -> dict:
    """Display assistant response."""
    if state.get("verbose", False):
        print("\n[NODE] output_node")
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            print(f"\nAssistant: {msg.content}")
            break
    return {}


def route_after_input(state: ConversationState) -> Literal["call_model", "end", "input"]:
    cmd = state.get("command")
    if cmd == "exit":
        return "end"
    if cmd in ["verbose", "quiet"]:
        return "input"
    return "call_model"


def route_after_model(state: ConversationState) -> Literal["tools", "output"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "output"


# ============================================
# GRAPH
# ============================================

def create_graph(checkpointer=None):
    workflow = StateGraph(ConversationState)
    workflow.add_node("input", input_node)
    workflow.add_node("call_model", call_model)
    workflow.add_node("tools", tools_node)
    workflow.add_node("output", output_node)

    workflow.set_entry_point("input")
    workflow.add_conditional_edges("input", route_after_input, {"call_model": "call_model", "input": "input", "end": END})
    workflow.add_conditional_edges("call_model", route_after_model, {"tools": "tools", "output": "output"})
    workflow.add_edge("tools", "call_model")
    workflow.add_edge("output", "input")

    return workflow.compile(checkpointer=checkpointer)


# ============================================
# MERMAID DIAGRAM (for portfolio)
# ============================================

def get_mermaid_diagram() -> str:
    """Mermaid diagram of the system."""
    return """flowchart TD
    START([START]) --> input
    input[input_node] -->|user message| call_model
    input -->|exit| END([END])
    input -->|verbose/quiet| input
    call_model[call_model] -->|has tool_calls| tools
    call_model -->|final answer| output
    tools[tools_node] --> call_model
    output[output_node] --> input"""


# ============================================
# MAIN
# ============================================

async def main():
    thread_id = os.environ.get("TASK5_THREAD_ID", "task5_conversation")
    config = {"configurable": {"thread_id": thread_id}} if CHECKPOINTER else {}
    initial_state = {"messages": [], "verbose": False, "command": None}

    print("=" * 70)
    print("Task 5: Persistent LangGraph Conversation with Checkpointing")
    print("=" * 70)
    print("\nMermaid diagram:")
    print(get_mermaid_diagram())
    print("\nCommands: quit/exit, verbose, quiet")
    print("Tools: calculator, count_letter, reverse_words")
    print("=" * 70)

    if CHECKPOINTER_TYPE == "sqlite" and HAS_SQLITE:
        db_path = os.environ.get("TASK5_DB", ".checkpoints/task5.db")
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        conn_str = f"file:{db_path}?mode=rwc"
        print(f"\n[SqliteSaver: persistent recovery to {db_path}]")
        print(f"  Thread ID: {thread_id}")
        print("  Resume: run again with same TASK5_THREAD_ID")
        with SqliteSaver.from_conn_string(conn_str) as sqlite_checkpointer:
            app = create_graph(sqlite_checkpointer)
            try:
                await app.ainvoke(initial_state, config=config)
            except KeyboardInterrupt:
                print("\n\nInterrupted. State saved. Run again to resume.")
    else:
        if CHECKPOINTER:
            print("\n[MemorySaver: checkpointing in-memory (state lost on exit)]")
        else:
            print("\n[No checkpointer - install langgraph with checkpoint support]")
        app = create_graph(CHECKPOINTER)
        try:
            await app.ainvoke(initial_state, config=config)
        except KeyboardInterrupt:
            print("\n\nInterrupted.")

    print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main())
