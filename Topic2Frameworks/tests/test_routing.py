"""Unit tests for routing logic (Task 5, 7, 8): 'Hey Qwen' -> Qwen, else -> Llama."""

import pytest


def route_to_model(user_input: str) -> str:
    """Pure routing logic: 'Hey Qwen' -> Qwen, else -> Llama."""
    if not user_input or not user_input.strip():
        return "empty"
    if user_input.strip().lower().startswith("hey qwen"):
        return "qwen"
    return "llama"


class TestRouting:
    """Tests for selective routing between Llama and Qwen."""

    def test_hey_qwen_routes_to_qwen(self):
        assert route_to_model("Hey Qwen, what is 2+2?") == "qwen"
        assert route_to_model("hey qwen") == "qwen"
        assert route_to_model("  Hey Qwen  ") == "qwen"

    def test_other_input_routes_to_llama(self):
        assert route_to_model("What is 2+2?") == "llama"
        assert route_to_model("Hello") == "llama"
        assert route_to_model("Hey Llama") == "llama"

    def test_empty_input(self):
        assert route_to_model("") == "empty"
        assert route_to_model("   ") == "empty"
