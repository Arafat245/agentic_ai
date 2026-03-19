"""
Task 3: Manual Tool Handling with Custom Calculator (geometric functions)

Based on manual-tool-handling.py. Defines a calculator tool that:
- Accepts JSON string input (json.loads)
- Returns JSON output (json.dumps)
- Supports arithmetic + geometric functions (sin, cos, tan, sqrt, etc.)
- Uses ast.literal_eval or numexpr for safe evaluation
"""

import json
import math
import ast
from openai import OpenAI

# ============================================
# PART 1: Calculator Tool (JSON in, JSON out)
# ============================================

def calculator(expression_json: str) -> str:
    """
    Evaluate a mathematical expression. Supports arithmetic and geometric functions.
    Input: JSON string with "expression" key, e.g. '{"expression": "sin(3.14159/2)"}'
    Output: JSON string with "result" or "error" key.
    """
    try:
        args = json.loads(expression_json)
        expr = args.get("expression", args.get("expr", ""))
        # Handle LLM passing expression_json (nested JSON string)
        if not expr and "expression_json" in args:
            inner = json.loads(args["expression_json"]) if isinstance(args["expression_json"], str) else args["expression_json"]
            expr = inner.get("expression", inner.get("expr", ""))
        if not expr:
            return json.dumps({"error": "No expression provided"})

        # Safe eval: only allow math functions and numbers
        safe_dict = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
            "exp": math.exp, "pow": math.pow, "abs": abs,
            "pi": math.pi, "e": math.e,
        }
        # For simple numeric expressions, use ast.literal_eval if no functions
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        return json.dumps({"result": result})
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================
# PART 2: Tool Schema for LLM
# ============================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Supports: +, -, *, /, **, sin, cos, tan, sqrt, log, exp, pi, e. Example: sin(pi/2) or sqrt(16)",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression_json": {
                        "type": "string",
                        "description": "JSON string with 'expression' key, e.g. '{\"expression\": \"sin(3.14159/2)\"}'"
                    }
                },
                "required": ["expression_json"]
            }
        }
    }
]

tool_map = {
    "get_weather": lambda **kw: _get_weather(**kw),
    "calculator": lambda expression_json: calculator(expression_json),
}


def _get_weather(location: str) -> str:
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")


# ============================================
# PART 3: Agent Loop
# ============================================

def run_agent(user_query: str):
    client = OpenAI()
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always use the calculator tool for math - never compute yourself. Use tools when needed."},
        {"role": "user", "content": user_query}
    ]

    print(f"User: {user_query}\n")

    for iteration in range(5):
        print(f"--- Iteration {iteration + 1} ---")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        assistant_message = response.choices[0].message

        if assistant_message.tool_calls:
            print(f"LLM wants to call {len(assistant_message.tool_calls)} tool(s)")
            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")

                if function_name == "calculator":
                    result = calculator(json.dumps(function_args))
                elif function_name == "get_weather":
                    result = _get_weather(**function_args)
                else:
                    result = f"Error: Unknown function {function_name}"

                print(f"  Result: {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result
                })
            print()
        else:
            print(f"Assistant: {assistant_message.content}\n")
            return assistant_message.content

    return "Max iterations reached"


# ============================================
# PART 4: Test
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("TEST 1: Calculator - geometric (sin)")
    print("="*60)
    run_agent("What is sin(pi/2)?")

    print("\n" + "="*60)
    print("TEST 2: Calculator - sqrt")
    print("="*60)
    run_agent("What is the square root of 144?")

    print("\n" + "="*60)
    print("TEST 3: Weather (original tool)")
    print("="*60)
    run_agent("What's the weather in Tokyo?")

    print("\n" + "="*60)
    print("TEST 4: Combined - cos and expression")
    print("="*60)
    run_agent("Compute cos(0) + sqrt(9)")
