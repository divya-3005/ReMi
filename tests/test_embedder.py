"""
tests/test_embedder.py
──────────────────────
Phase 4 test suite: GeminiEmbedder (google-genai SDK >= 2.12).

Tests batching, retry logic, and graceful degradation (zero-vector fallback)
without making any real network calls.

Key mock targets:
  - `src.vectorstore.embedder.genai.Client` — the Client constructor
  - `mock_client.models.embed_content` — the actual API call
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.api_core import exceptions as google_exc

from src.vectorstore.embedder import GeminiEmbedder
from src.vectorstore.errors import EmbeddingError


def _make_response(vectors: list[list[float]]) -> MagicMock:
    """
    Build a mock EmbedContentResponse where response.embeddings[i].values == vectors[i].
    """
    response = MagicMock()
    response.embeddings = [SimpleNamespace(values=v) for v in vectors]
    return response


@pytest.fixture
def mock_client():
    """Return a mock genai.Client instance."""
    return MagicMock()


@pytest.fixture
def embedder(mock_settings, mock_client):
    """Return a GeminiEmbedder with the genai.Client patched to mock_client."""
    with patch("src.vectorstore.embedder.genai.Client", return_value=mock_client):
        emb = GeminiEmbedder(mock_settings)
    # Replace the _client with mock_client directly to control calls
    emb._client = mock_client
    return emb


# ── embed_query ───────────────────────────────────────────────────────────────

class TestEmbedQuery:
    def test_success_returns_list_of_floats(self, embedder, mock_client, mock_settings):
        mock_client.models.embed_content.return_value = _make_response([[0.1, 0.2, 0.3]])

        result = embedder.embed_query("What is the GDP?")

        assert result == [0.1, 0.2, 0.3]

        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        assert call_kwargs["model"] == mock_settings.gemini_embedding_model
        assert call_kwargs["contents"] == "What is the GDP?"
        assert call_kwargs["config"].task_type == "RETRIEVAL_QUERY"

    def test_retry_on_resource_exhausted(self, embedder, mock_client):
        mock_client.models.embed_content.side_effect = [
            google_exc.ResourceExhausted("Rate limit exceeded"),
            google_exc.ResourceExhausted("Rate limit exceeded"),
            _make_response([[0.5, 0.5]]),
        ]

        with patch("src.vectorstore.embedder.time.sleep"):  # don't actually sleep
            result = embedder.embed_query("Test query")

        assert result == [0.5, 0.5]
        assert mock_client.models.embed_content.call_count == 3

    def test_raises_error_after_max_retries(self, embedder, mock_client):
        mock_client.models.embed_content.side_effect = google_exc.ResourceExhausted(
            "Rate limit exceeded"
        )

        with patch("src.vectorstore.embedder.time.sleep"):
            with pytest.raises(EmbeddingError) as exc_info:
                embedder.embed_query("Test query")

        assert "retries" in str(exc_info.value).lower()
        # 1 initial + 3 retries = 4 total calls
        assert mock_client.models.embed_content.call_count == 4

    def test_non_rate_limit_error_fails_immediately(self, embedder, mock_client):
        mock_client.models.embed_content.side_effect = RuntimeError("Auth failed")

        with pytest.raises(EmbeddingError):
            embedder.embed_query("Test query")

        # Must NOT retry on non-rate-limit errors
        assert mock_client.models.embed_content.call_count == 1


# ── embed_texts ───────────────────────────────────────────────────────────────

class TestEmbedTexts:
    def test_empty_list_returns_empty(self, embedder, mock_client):
        result = embedder.embed_texts([])
        assert result == []
        mock_client.models.embed_content.assert_not_called()

    def test_success_returns_list_of_embeddings(self, embedder, mock_client, mock_settings):
        mock_client.models.embed_content.return_value = _make_response(
            [[0.1, 0.1], [0.2, 0.2]]
        )

        texts = ["Text 1", "Text 2"]
        result = embedder.embed_texts(texts)

        assert len(result) == 2
        assert result[0] == [0.1, 0.1]
        assert result[1] == [0.2, 0.2]

        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        assert call_kwargs["contents"] == texts
        assert call_kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"

    def test_batches_large_requests(self, embedder, mock_client, mock_settings):
        embedder.BATCH_SIZE = 100

        def side_effect(model, contents, config):
            dim = mock_settings.embedding_dim
            return _make_response([[0.0] * dim for _ in contents])

        mock_client.models.embed_content.side_effect = side_effect

        texts = [f"Text {i}" for i in range(250)]
        result = embedder.embed_texts(texts)

        assert len(result) == 250
        assert mock_client.models.embed_content.call_count == 3  # 100 + 100 + 50

        calls = mock_client.models.embed_content.call_args_list
        assert len(calls[0].kwargs["contents"]) == 100
        assert len(calls[1].kwargs["contents"]) == 100
        assert len(calls[2].kwargs["contents"]) == 50

    def test_graceful_degradation_inserts_zero_vectors(
        self, embedder, mock_client, mock_settings
    ):
        embedder.BATCH_SIZE = 2
        dim = mock_settings.embedding_dim

        def side_effect(model, contents, config):
            if "A" in contents:
                return _make_response([[1.0] * dim, [1.0] * dim])
            raise google_exc.ResourceExhausted("Rate limit exceeded")

        mock_client.models.embed_content.side_effect = side_effect

        with patch("src.vectorstore.embedder.time.sleep"):
            result = embedder.embed_texts(["A", "B", "C", "D"])

        assert len(result) == 4
        # Batch 0 (A, B) succeeded
        assert result[0] == [1.0] * dim
        assert result[1] == [1.0] * dim
        # Batch 1 (C, D) failed → zero-vectors
        assert result[2] == [0.0] * dim
        assert result[3] == [0.0] * dim
