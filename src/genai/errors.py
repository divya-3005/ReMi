"""
src/genai/errors.py
───────────────────
Exceptions for the LLM client layer.
"""

from __future__ import annotations


class LLMError(Exception):
    """
    Raised when an LLM call fails after all retries are exhausted.
    Wraps the original exception as __cause__ for full traceback visibility.
    """
