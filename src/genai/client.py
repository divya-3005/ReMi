"""
src/genai/client.py
───────────────────
LLM client wrappers for Gemini (structured + text generation)
and Groq (fast inference for the analyzer step).

Design:
- Model names are sourced ONLY from Settings — never hardcoded here.
- Both clients expose identical interfaces: generate() and generate_text(),
  making them drop-in swappable in agent code.
- Both clients have exponential backoff retry (×3) and raise LLMError on
  exhaustion — callers get a named exception, not a raw SDK exception.
- generate() handles markdown fence stripping (LLMs often wrap JSON
  in ```json...```) before passing to Pydantic for validation.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Type, TypeVar

import google.genai as genai
import google.genai.types as genai_types
from groq import Groq
from pydantic import BaseModel, ValidationError

from src.config import Settings
from src.genai.errors import LLMError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MAX_RETRIES = 3
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _strip_json_fences(text: str) -> str:
    """
    Strip markdown code fences from LLM output.
    LLMs often wrap JSON in ```json ... ``` even when instructed not to.
    Extracts the content inside the first fence if one is found.
    """
    match = _JSON_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _retry_with_backoff(fn, max_retries: int = MAX_RETRIES, label: str = "LLM call"):
    """
    Call fn(), retrying up to max_retries times with exponential backoff.
    Raises LLMError if all attempts fail.
    """
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt == max_retries:
                break
            logger.warning(
                f"{label} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
            delay *= 2
    raise LLMError(
        f"{label} failed after {max_retries} retries: {last_exc}"
    ) from last_exc


class GeminiClient:
    """
    Thin wrapper around the Gemini LLM API (google-genai SDK).

    generate()      → structured JSON output parsed into a Pydantic model
    generate_text() → plain text output (for synthesis / open-ended generation)
    """

    def __init__(self, settings: Settings):
        # Model name sourced from Settings — only place it lives.
        self.model_name = settings.gemini_llm_model
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def generate(self, prompt: str, response_schema: Type[T]) -> T:
        """
        Call Gemini and parse the response into a Pydantic model.

        Uses JSON mode (response_mime_type="application/json") where possible,
        but also handles markdown fence stripping for models that ignore the
        mime type hint.

        Args:
            prompt: The full prompt string.
            response_schema: A Pydantic BaseModel subclass to parse the response into.

        Returns:
            An instance of response_schema.

        Raises:
            LLMError: If the API call fails after all retries.
        """
        def _call():
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            raw = _strip_json_fences(response.text)
            data = json.loads(raw)
            return response_schema.model_validate(data)

        return _retry_with_backoff(
            _call,
            label=f"GeminiClient.generate({response_schema.__name__})"
        )

    def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Call Gemini and return the raw text response.

        Args:
            prompt: The full prompt string.
            temperature: Sampling temperature (higher = more creative).

        Returns:
            The text of the first candidate.

        Raises:
            LLMError: If the API call fails after all retries.
        """
        def _call():
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=temperature,
                ),
            )
            return response.text

        return _retry_with_backoff(_call, label="GeminiClient.generate_text")


class GroqClient:
    """
    Thin wrapper around the Groq inference API.

    Used for the AnalyzerAgent where fast, cheap inference matters more than
    frontier model quality. Interface is identical to GeminiClient.
    """

    def __init__(self, settings: Settings):
        self.model_name = settings.groq_llm_model
        self._client = Groq(api_key=settings.groq_api_key)

    def generate(self, prompt: str, response_schema: Type[T]) -> T:
        """
        Call Groq and parse the response into a Pydantic model.

        Args:
            prompt: The full prompt string.
            response_schema: A Pydantic BaseModel subclass to parse into.

        Returns:
            An instance of response_schema.

        Raises:
            LLMError: If the API call fails after all retries.
        """
        def _call():
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = _strip_json_fences(response.choices[0].message.content)
            data = json.loads(raw)
            return response_schema.model_validate(data)

        return _retry_with_backoff(
            _call,
            label=f"GroqClient.generate({response_schema.__name__})"
        )

    def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Call Groq and return the raw text response.

        Raises:
            LLMError: If the API call fails after all retries.
        """
        def _call():
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content

        return _retry_with_backoff(_call, label="GroqClient.generate_text")
