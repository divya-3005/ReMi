"""
tests/test_clients.py
─────────────────────
Phase 6 test suite: GeminiClient and GroqClient.

All external API calls are mocked. Tests verify:
- Structured JSON generation + Pydantic parse
- Plain text generation
- Exponential backoff retry logic
- LLMError raised after max retries exhausted
- Model name sourced from Settings, never hardcoded
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from pydantic import BaseModel

from src.genai.client import GeminiClient, GroqClient
from src.genai.errors import LLMError


# ── Test schema ───────────────────────────────────────────────────────────────

class _TestSchema(BaseModel):
    name: str
    value: int


# ── GeminiClient fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def mock_gemini_sdk_client():
    return MagicMock()


@pytest.fixture
def gemini_client(mock_settings, mock_gemini_sdk_client):
    with patch("src.genai.client.genai.Client", return_value=mock_gemini_sdk_client):
        client = GeminiClient(mock_settings)
    client._client = mock_gemini_sdk_client
    return client


def _make_gemini_text_response(text: str) -> MagicMock:
    """Return a mock SDK GenerateContentResponse with .text attribute."""
    resp = MagicMock()
    resp.text = text
    return resp


# ── GeminiClient tests ────────────────────────────────────────────────────────

class TestGeminiClient:
    def test_model_name_comes_from_settings(self, gemini_client, mock_settings):
        assert gemini_client.model_name == mock_settings.gemini_llm_model

    def test_generate_text_returns_string(self, gemini_client, mock_gemini_sdk_client):
        mock_gemini_sdk_client.models.generate_content.return_value = (
            _make_gemini_text_response("Hello world")
        )
        result = gemini_client.generate_text("What is 2+2?")
        assert result == "Hello world"

    def test_generate_calls_correct_model(self, gemini_client, mock_gemini_sdk_client, mock_settings):
        mock_gemini_sdk_client.models.generate_content.return_value = (
            _make_gemini_text_response(json.dumps({"name": "test", "value": 42}))
        )
        gemini_client.generate("Prompt", _TestSchema)
        call_kwargs = mock_gemini_sdk_client.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == mock_settings.gemini_llm_model

    def test_generate_parses_json_into_pydantic_model(self, gemini_client, mock_gemini_sdk_client):
        mock_gemini_sdk_client.models.generate_content.return_value = (
            _make_gemini_text_response(json.dumps({"name": "Alice", "value": 7}))
        )
        result = gemini_client.generate("Prompt", _TestSchema)
        assert isinstance(result, _TestSchema)
        assert result.name == "Alice"
        assert result.value == 7

    def test_generate_strips_markdown_fences_before_parse(self, gemini_client, mock_gemini_sdk_client):
        """LLMs often wrap JSON in ```json ... ``` — we must strip that."""
        payload = "```json\n" + json.dumps({"name": "Bob", "value": 3}) + "\n```"
        mock_gemini_sdk_client.models.generate_content.return_value = (
            _make_gemini_text_response(payload)
        )
        result = gemini_client.generate("Prompt", _TestSchema)
        assert result.name == "Bob"

    def test_generate_retries_on_exception(self, gemini_client, mock_gemini_sdk_client):
        mock_gemini_sdk_client.models.generate_content.side_effect = [
            RuntimeError("Transient error"),
            RuntimeError("Transient error"),
            _make_gemini_text_response(json.dumps({"name": "OK", "value": 1})),
        ]
        with patch("src.genai.client.time.sleep"):
            result = gemini_client.generate("Prompt", _TestSchema)
        assert result.name == "OK"
        assert mock_gemini_sdk_client.models.generate_content.call_count == 3

    def test_generate_raises_llm_error_after_max_retries(self, gemini_client, mock_gemini_sdk_client):
        mock_gemini_sdk_client.models.generate_content.side_effect = RuntimeError("Persistent error")
        with patch("src.genai.client.time.sleep"):
            with pytest.raises(LLMError) as exc_info:
                gemini_client.generate("Prompt", _TestSchema)
        assert "retries" in str(exc_info.value).lower()

    def test_generate_text_retries_on_exception(self, gemini_client, mock_gemini_sdk_client):
        mock_gemini_sdk_client.models.generate_content.side_effect = [
            RuntimeError("fail"),
            _make_gemini_text_response("success"),
        ]
        with patch("src.genai.client.time.sleep"):
            result = gemini_client.generate_text("Prompt")
        assert result == "success"

    def test_generate_text_raises_llm_error_after_max_retries(self, gemini_client, mock_gemini_sdk_client):
        mock_gemini_sdk_client.models.generate_content.side_effect = RuntimeError("fail")
        with patch("src.genai.client.time.sleep"):
            with pytest.raises(LLMError):
                gemini_client.generate_text("Prompt")


# ── GroqClient fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def mock_groq_sdk_client():
    return MagicMock()


@pytest.fixture
def groq_client(mock_settings, mock_groq_sdk_client):
    with patch("src.genai.client.Groq", return_value=mock_groq_sdk_client):
        client = GroqClient(mock_settings)
    client._client = mock_groq_sdk_client
    return client


def _make_groq_response(text: str) -> MagicMock:
    """Return a mock Groq ChatCompletion response."""
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── GroqClient tests ──────────────────────────────────────────────────────────

class TestGroqClient:
    def test_model_name_comes_from_settings(self, groq_client, mock_settings):
        assert groq_client.model_name == mock_settings.groq_llm_model

    def test_generate_text_returns_string(self, groq_client, mock_groq_sdk_client):
        mock_groq_sdk_client.chat.completions.create.return_value = (
            _make_groq_response("Groq response")
        )
        result = groq_client.generate_text("Some prompt")
        assert result == "Groq response"

    def test_generate_parses_json_into_pydantic_model(self, groq_client, mock_groq_sdk_client):
        mock_groq_sdk_client.chat.completions.create.return_value = (
            _make_groq_response(json.dumps({"name": "Carol", "value": 99}))
        )
        result = groq_client.generate("Prompt", _TestSchema)
        assert isinstance(result, _TestSchema)
        assert result.name == "Carol"

    def test_groq_retries_on_exception(self, groq_client, mock_groq_sdk_client):
        mock_groq_sdk_client.chat.completions.create.side_effect = [
            RuntimeError("Rate limit"),
            _make_groq_response(json.dumps({"name": "retry_ok", "value": 0})),
        ]
        with patch("src.genai.client.time.sleep"):
            result = groq_client.generate("Prompt", _TestSchema)
        assert result.name == "retry_ok"

    def test_groq_raises_llm_error_after_max_retries(self, groq_client, mock_groq_sdk_client):
        mock_groq_sdk_client.chat.completions.create.side_effect = RuntimeError("fail")
        with patch("src.genai.client.time.sleep"):
            with pytest.raises(LLMError):
                groq_client.generate("Prompt", _TestSchema)
