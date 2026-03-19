"""
Task 4: LangGraph Agent with Multiple Tools

Tools:
1. Calculator (from Task 3) - arithmetic + geometric
2. Letter count - count occurrences of a letter in text
3. Word reverse - custom tool (reverses each word in a string)

Uses tool_map for clean dispatch (no long if/else).
"""

import json
import math
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# ============================================
# TOOL 1: Calculator (from Task 3)
# ============================================

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression. Supports: +, -, *, /, **, sin, cos, tan, sqrt, log, exp, pi, e.
    Example: sin(pi/2), sqrt(16), 2+3*4
    """
    try:
        # Support both raw expression and JSON-wrapped (assignment: json.loads)
        try:
            args = json.loads(expression)
            expr = args.get("expression", args.get("expr", expression))
        except json.JSONDecodeError:
            expr = expression
        if not expr:
            return json.dumps({"error": "No expression provided"})
        safe_dict = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
            "exp": math.exp, "pow": math.pow, "abs": abs,
            "pi": math.pi, "e": math.e,
        }
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================
# TOOL 2: Letter count
# ============================================

@tool
def count_letter(text: str, letter: str) -> str:
    """
    Count the number of occurrences of a letter (case-insensitive) in a piece of text.
    Example: count_letter("Mississippi riverboats", "s") returns count of 's'.
    """
    if len(letter) != 1:
        return json.dumps({"error": "letter must be a single character"})
    count = text.lower().count(letter.lower())
    return json.dumps({"text": text, "letter": letter, "count": count})


# ============================================
# TOOL 3: Word reverse (custom)
# ============================================

@tool
def reverse_words(text: str) -> str:
    """
    Reverse each word in the given text. Words are separated by spaces.
    Example: "hello world" -> "olleh dlrow"
    """
    words = text.split()
    reversed_words = [w[::-1] for w in words]
    return json.dumps({"original": text, "result": " ".join(reversed_words)})


# ============================================
# Tool list and map
# ============================================

tools = [calculator, count_letter, reverse_words]
tool_map = {t.name: t for t in tools}

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)


def run_agent(user_query: str):
    messages = [
        SystemMessage(content="You are a helpful assistant. Use the provided tools when needed. For math, use the calculator. For counting letters, use count_letter."),
        HumanMessage(content=user_query)
    ]

    print(f"User: {user_query}\n")

    for iteration in range(5):
        print(f"--- Iteration {iteration + 1} ---")

        response = llm_with_tools.invoke(messages)

        if response.tool_calls:
            print(f"LLM wants to call {len(response.tool_calls)} tool(s)")
            messages.append(response)

            for tool_call in response.tool_calls:
                function_name = tool_call["name"]
                function_args = tool_call["args"]

                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")

                if function_name in tool_map:
                    result = tool_map[function_name].invoke(function_args)
                else:
                    result = f"Error: Unknown function {function_name}"

                print(f"  Result: {result}")

                messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
            print()
        else:
            print(f"Assistant: {response.content}\n")
            return response.content

    return "Max iterations reached"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your question: ")
    run_agent(query)
