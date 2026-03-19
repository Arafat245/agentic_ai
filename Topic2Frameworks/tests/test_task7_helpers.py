"""Unit tests for Task 7 multi-LLM chat helpers: format and clean functions."""

import pytest

try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from task7_multi_llm_history import _format_for_llama, _format_for_qwen, _clean_response
except ImportError:
    HumanMessage = AIMessage = SystemMessage = None
    _format_for_llama = _format_for_qwen = _clean_response = None

pytestmark = pytest.mark.skipif(
    _format_for_llama is None,
    reason="task7 deps (transformers, langchain-huggingface) not installed",
)


class TestFormatForLlama:
    """Tests for _format_for_llama: Human=user, Llama=assistant, Qwen=user."""

    def test_human_message(self):
        msgs = [HumanMessage(content="Human: Hello")]
        out = _format_for_llama(msgs)
        assert out == [{"role": "user", "content": "Human: Hello"}]

    def test_llama_reply_as_assistant(self):
        msgs = [AIMessage(content="Llama: Hi there")]
        out = _format_for_llama(msgs)
        assert out == [{"role": "assistant", "content": "Llama: Hi there"}]

    def test_qwen_reply_as_user(self):
        msgs = [AIMessage(content="Qwen: Chocolate is best")]
        out = _format_for_llama(msgs)
        assert out == [{"role": "user", "content": "Qwen: Chocolate is best"}]

    def test_mixed_conversation(self):
        msgs = [
            HumanMessage(content="Human: What is best?"),
            AIMessage(content="Llama: Vanilla."),
            HumanMessage(content="Human: Hey Qwen, what do you think?"),
            AIMessage(content="Qwen: Chocolate!"),
        ]
        out = _format_for_llama(msgs)
        assert out[0] == {"role": "user", "content": "Human: What is best?"}
        assert out[1] == {"role": "assistant", "content": "Llama: Vanilla."}
        assert out[2] == {"role": "user", "content": "Human: Hey Qwen, what do you think?"}
        assert out[3] == {"role": "user", "content": "Qwen: Chocolate!"}


class TestFormatForQwen:
    """Tests for _format_for_qwen: Human=user, Llama=user, Qwen=assistant."""

    def test_qwen_reply_as_assistant(self):
        msgs = [AIMessage(content="Qwen: Chocolate!")]
        out = _format_for_qwen(msgs)
        assert out == [{"role": "assistant", "content": "Qwen: Chocolate!"}]

    def test_llama_reply_as_user(self):
        msgs = [AIMessage(content="Llama: Vanilla.")]
        out = _format_for_qwen(msgs)
        assert out == [{"role": "user", "content": "Llama: Vanilla."}]


class TestCleanResponse:
    """Tests for _clean_response: strip model artifacts."""

    def test_strips_llama_artifacts(self):
        text = "<|start_header_id|>assistant<|end_header_id|>Llama: Hello<|eot_id|>"
        assert _clean_response(text) == "Llama: Hello"

    def test_strips_prompt_prefix(self):
        text = "system: You are helpful\nuser: Hi\nassistant: Hello"
        out = _clean_response(text, prompt_with_gen="system: You are helpful\nuser: Hi\nassistant: ")
        assert out == "Hello"

    def test_empty_unchanged(self):
        assert _clean_response("") == ""
        assert _clean_response("  ") == ""
