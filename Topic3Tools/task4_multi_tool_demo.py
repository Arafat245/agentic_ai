"""
Task 4: Multi-tool Queries Demo (LangGraph tool handling)

Runs predefined queries that demonstrate:
- Multiple tool invocations in one turn (e.g., count i's and s's)
- Chained tool use (count -> calculator)
- Query using all three tools
- Sequential chaining toward 5-turn limit

Uses the same agent as task4 but with batch test queries.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task4_langgraph_tools_basic import run_agent

# Predefined test queries for multi-tool demonstration
TEST_QUERIES = [
    ("Q1: Letter count twice", "How many s's are in Mississippi riverboats? And how many i's?"),
    ("Q2: Compare counts", "Are there more i's than s's in Mississippi riverboats?"),
    ("Q3: Sin of difference", "What is the sin of the difference between the number of i's and the number of s's in Mississippi riverboats?"),
    ("Q4: All three tools", "Take the text 'hello world', reverse each word, then count how many l's are in the result, and finally compute the square root of that count."),
    ("Q5: Chained (5 turns)", "First tell me how many s's in Mississippi. Then how many i's. Then compute their difference. Then take the square root. Finally reverse the words in 'done'."),
]


def main():
    print("="*70)
    print("Task 4: Multi-Tool Queries Demo (LangGraph)")
    print("="*70)

    for label, query in TEST_QUERIES:
        print("\n" + "="*70)
        print(label)
        print("="*70)
        run_agent(query)


if __name__ == "__main__":
    main()
