"""
src/vectorstore/errors.py
─────────────────────────
Exceptions for the vectorstore package.
"""

from __future__ import annotations


class VectorStoreError(Exception):
    """Base class for all vector store errors."""


class EmbeddingError(VectorStoreError):
    """Raised when the external embedding API fails."""
