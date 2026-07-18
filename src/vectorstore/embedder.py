"""
src/vectorstore/embedder.py
───────────────────────────
Client for the Google GenAI embeddings API (google-genai SDK >= 2.12).

Migrated from the deprecated `google.generativeai` package which reached
end-of-life in mid-2025. The new `google.genai` SDK uses a stateful `Client`
object rather than module-level globals, which is also safer in multi-threaded
contexts.

Key API differences from the old SDK:
  Old: genai.embed_content(model=..., content=..., task_type="...")
       response["embedding"]                    # list[float] for single query
  New: client.models.embed_content(
           model=..., contents=...,
           config=types.EmbedContentConfig(taskType="..."))
       response.embeddings[0].values            # list[float] for single
       response.embeddings[i].values            # list[float] for batch item i
"""

from __future__ import annotations

import logging
import time
from typing import List

import google.genai as genai
import google.genai.types as genai_types
from google.api_core import exceptions as google_exc

from src.config import Settings
from src.vectorstore.errors import EmbeddingError

logger = logging.getLogger(__name__)


class GeminiEmbedder:
    """
    Wrapper around the Gemini Embedding API (google-genai SDK).

    Includes automatic batching for large document uploads and exponential
    backoff for rate limits (ResourceExhausted).
    """

    BATCH_SIZE = 100
    MAX_RETRIES = 3

    def __init__(self, settings: Settings):
        self.model_name = settings.gemini_embedding_model
        self.dim = settings.embedding_dim
        # The new SDK uses a Client instance rather than module-level config.
        # This is thread-safe: Client is a thin stateless wrapper over the API key.
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def _call_api_with_retry(
        self,
        contents: str | List[str],
        task_type: str,
    ) -> genai_types.EmbedContentResponse:
        """
        Execute the embed_content call with exponential backoff.
        Raises EmbeddingError if all retries are exhausted.
        """
        config = genai_types.EmbedContentConfig(task_type=task_type)
        delay = 1.0

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                return self._client.models.embed_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
            except google_exc.ResourceExhausted as e:
                if attempt == self.MAX_RETRIES:
                    raise EmbeddingError(
                        f"Failed to embed content after {self.MAX_RETRIES} retries: {e}"
                    ) from e
                logger.warning(
                    f"Gemini API rate limit hit (attempt {attempt + 1}/{self.MAX_RETRIES}). "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                # Auth errors, invalid model names, etc. — fail immediately
                raise EmbeddingError(f"Unexpected embedding error: {e}") from e

        raise EmbeddingError("Failed to embed content after retries")  # unreachable

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single search query using the retrieval_query task type.

        Returns:
            A list of floats representing the embedding vector.
        """
        response = self._call_api_with_retry(query, "RETRIEVAL_QUERY")
        if response.embeddings and response.embeddings[0].values:
            return [float(v) for v in response.embeddings[0].values]
        return []

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document chunks using the retrieval_document task type.
        Handles batching automatically.

        If a batch fails completely (all retries exhausted), it gracefully degrades
        by inserting zero-vectors rather than crashing the entire ingestion process.
        Affected chunks will not be searchable via dense vector search.

        Returns:
            A list of embedding vectors (one per input text), same order as input.
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            try:
                response = self._call_api_with_retry(batch, "RETRIEVAL_DOCUMENT")
                # response.embeddings[j].values is the embedding for batch[j]
                if response.embeddings:
                    for emb in response.embeddings:
                        if emb.values:
                            all_embeddings.append([float(v) for v in emb.values])
                        else:
                            all_embeddings.append([0.0] * self.dim)
            except EmbeddingError as e:
                logger.error(
                    f"Batch {i // self.BATCH_SIZE} (items {i}–{i + len(batch) - 1}) "
                    f"failed completely: {e}. "
                    "Gracefully degrading: inserting zero-vectors for this batch. "
                    "These chunks will not be searchable via dense vector search."
                )
                zero_vector = [0.0] * self.dim
                all_embeddings.extend([zero_vector] * len(batch))

        return all_embeddings
