"""
tests/conftest.py
─────────────────
Shared pytest fixtures available to all test modules.

Key design: tests never touch real API keys or real external services.
The `mock_settings` fixture overrides all API credentials with sentinel
values and exposes a predictable Settings object. Tests that need to verify
Settings construction from environment variables should call
get_settings.cache_clear() in their setup and restore afterward.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings, get_settings


# ── Settings fixture ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_settings() -> Settings:
    """
    Return a Settings instance with test-safe sentinel values.

    Uses object construction directly (not environment variables) so tests
    are hermetic — no risk of a real .env leaking into the test process.
    """
    # Clear the lru_cache so this fixture always produces a fresh instance
    get_settings.cache_clear()
    return Settings(
        gemini_api_key="test-gemini-key",
        groq_api_key="test-groq-key",
        gemini_llm_model="gemini-2.5-flash",
        gemini_embedding_model="gemini-embedding-001",
        groq_llm_model="llama-3.1-8b-instant",
        chunk_size=500,
        chunk_overlap=100,
        retrieval_k=5,
        rrf_k_constant=60,
        embedding_dim=768,
        analyzer_relevance_floor=0.40,
        min_citation_coverage=0.55,
        min_citation_utilization=0.30,
        max_workflow_retries=1,
        research_timeout_seconds=30,
    )


# ── Text fixtures ─────────────────────────────────────────────────────────────

SAMPLE_TEXT = """\
Artificial intelligence is transforming industries worldwide. Machine learning
algorithms process vast amounts of data to extract meaningful patterns. Deep
learning, a subset of machine learning, uses neural networks with many layers.

Natural language processing enables computers to understand human language.
Large language models have demonstrated remarkable capabilities in text
generation, summarization, and question answering.

Vector databases store high-dimensional embeddings for semantic similarity
search. Retrieval-augmented generation combines document retrieval with
language model generation to produce grounded, accurate responses.
"""

SHORT_TEXT = "This is a short document with minimal content for edge case testing."


@pytest.fixture
def sample_text() -> str:
    """A multi-paragraph English text suitable for chunking and embedding tests."""
    return SAMPLE_TEXT


@pytest.fixture
def short_text() -> str:
    """A text shorter than any reasonable chunk_size, for boundary tests."""
    return SHORT_TEXT


# ── File path fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_txt_path(tmp_path) -> str:
    """Write SAMPLE_TEXT to a real temp .txt file and return the path."""
    p = tmp_path / "sample.txt"
    p.write_text(SAMPLE_TEXT, encoding="utf-8")
    return str(p)


@pytest.fixture
def nonexistent_path(tmp_path) -> str:
    """Return a path that is guaranteed not to exist."""
    return str(tmp_path / "does_not_exist.pdf")


@pytest.fixture
def unsupported_path(tmp_path) -> str:
    """Return a path to a .docx file (unsupported type)."""
    p = tmp_path / "document.docx"
    p.write_bytes(b"PK\x03\x04")  # ZIP magic bytes; content doesn't matter
    return str(p)


# ── Embedding fixture ─────────────────────────────────────────────────────────

def make_zero_embedding(dim: int = 768) -> list[float]:
    """Return a zero vector of the given dimension."""
    return [0.0] * dim


def make_unit_embedding(dim: int = 768) -> list[float]:
    """Return a unit-norm-ish vector (first component 1.0, rest 0.0)."""
    v = [0.0] * dim
    v[0] = 1.0
    return v


@pytest.fixture
def mock_embedder(mock_settings):
    """
    A MagicMock that mimics GeminiEmbedder.

    embed_texts(texts) returns a list of zero-vectors.
    embed_query(query) returns a single zero-vector.
    Both use the embedding_dim from mock_settings.
    """
    embedder = MagicMock()
    dim = mock_settings.embedding_dim

    def _embed_texts(texts):
        return [make_zero_embedding(dim) for _ in texts]

    def _embed_query(query):
        return make_zero_embedding(dim)

    embedder.embed_texts.side_effect = _embed_texts
    embedder.embed_query.side_effect = _embed_query
    embedder.dim = dim
    return embedder
