"""
src/vectorstore/store.py
────────────────────────
Hybrid Vector Store combining FAISS (dense) and BM25 (sparse) with RRF fusion.

Thread safety:
A single re-entrant lock (`RLock`) guards all operations (read and write).
While FAISS `IndexFlatIP.search` is thread-safe for concurrent reads, it does
not guarantee safety if a read races against an `add_chunks` call that triggers
an internal buffer resize. To guarantee correctness without making assumptions
about underlying C++ buffer management, reads and writes share the same lock.
"""

from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.config import Settings
from src.models.schemas import Chunk, RetrievedContext
from src.vectorstore.embedder import GeminiEmbedder


class HybridStore:
    def __init__(self, embedder: GeminiEmbedder, settings: Settings):
        self.embedder = embedder
        self.dim = settings.embedding_dim
        self.rrf_k = settings.rrf_k_constant
        
        # Thread safety lock (re-entrant so load() can call clear/add)
        self._lock = threading.RLock()
        
        # Internal state
        self._init_empty_state()

    def _init_empty_state(self):
        """Initialize or reset all internal indices."""
        # FAISS IndexFlatIP uses Inner Product (cosine sim if vectors are normalized).
        # We don't normalize here because Gemini embeddings are generally close to
        # unit norm, and we rely on relative ranking anyway.
        self.faiss_index = faiss.IndexFlatIP(self.dim)
        
        self.chunks: List[Chunk] = []  # parallel to faiss_index (0-indexed)
        self.bm25: BM25Okapi | None = None
        
        # We need tokenized texts for BM25. We cache them so we don't re-tokenize
        # the entire corpus every time we add a small batch.
        self._tokenized_corpus: List[List[str]] = []

    def count(self) -> int:
        """Return total number of chunks in the store."""
        with self._lock:
            return len(self.chunks)

    def clear(self) -> None:
        """Remove all data from the store."""
        with self._lock:
            self._init_empty_state()

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace/punctuation tokenizer for BM25."""
        # This is crude compared to spaCy, but fast and sufficient for BM25
        # keyword overlap matching.
        import string
        text = text.lower()
        for p in string.punctuation:
            text = text.replace(p, " ")
        return [t for t in text.split() if t]

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Embed and index a list of chunks.
        Updates both FAISS and BM25 indices.
        """
        if not chunks:
            return

        # 1. Generate embeddings (API call, happens OUTSIDE the lock to not block reads)
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        
        # Convert to numpy float32 matrix (required by FAISS)
        emb_matrix = np.array(embeddings, dtype=np.float32)

        # Tokenize for BM25 (also outside the lock)
        tokenized_batch = [self._tokenize(t) for t in texts]

        # 2. Update indices (INSIDE the lock)
        with self._lock:
            # Dynamically set dimension if this is the first insert
            actual_dim = emb_matrix.shape[1]
            if not self.chunks and self.dim != actual_dim:
                self.dim = actual_dim
                self.faiss_index = faiss.IndexFlatIP(self.dim)

            # Add to FAISS
            self.faiss_index.add(emb_matrix)
            
            # Add to registry
            self.chunks.extend(chunks)
            
            # Rebuild BM25
            self._tokenized_corpus.extend(tokenized_batch)
            self.bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, k: int = 10) -> List[RetrievedContext]:
        """
        Search using Reciprocal Rank Fusion (FAISS + BM25).

        Design: embed_query() makes a network call and must NOT run while
        holding the lock. We do a lightweight empty-check first, then embed,
        then re-acquire the lock for all in-memory FAISS/BM25 operations.
        """
        # Fast path: empty store (no lock needed for a len() read on a Python list)
        if not self.chunks:
            return []

        # embed_query makes a network call — MUST be outside the lock.
        query_emb = np.array([self.embedder.embed_query(query)], dtype=np.float32)

        with self._lock:
            # Re-check inside lock: a clear() may have run between the fast path
            # and the lock acquisition.
            if not self.chunks:
                return []

            actual_k = min(k, len(self.chunks))
            dense_k = min(k * 2, len(self.chunks))

            # D = distances (scores), I = indices
            D, I = self.faiss_index.search(query_emb, dense_k)
            dense_results = {idx: rank for rank, idx in enumerate(I[0]) if idx != -1}
            
            # 2. Sparse search (BM25)
            sparse_results = {}
            if self.bm25:
                query_tokens = self._tokenize(query)
                sparse_scores = self.bm25.get_scores(query_tokens)
                
                # Get indices of top dense_k sparse scores
                # argsort sorts ascending, so take last dense_k and reverse
                top_sparse_idx = np.argsort(sparse_scores)[-dense_k:][::-1]
                
                for rank, idx in enumerate(top_sparse_idx):
                    if sparse_scores[idx] > 0:  # Only count if there's actual overlap
                        sparse_results[idx] = rank

            # 3. Reciprocal Rank Fusion (RRF)
            # RRF Score = 1 / (k + rank)
            rrf_scores = {}
            all_indices = set(dense_results.keys()) | set(sparse_results.keys())
            
            for idx in all_indices:
                score = 0.0
                if idx in dense_results:
                    score += 1.0 / (self.rrf_k + dense_results[idx])
                if idx in sparse_results:
                    score += 1.0 / (self.rrf_k + sparse_results[idx])
                rrf_scores[idx] = score

            # 4. Sort by RRF score and build result objects
            sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
            top_indices = sorted_indices[:actual_k]
            
            results = []
            for idx in top_indices:
                results.append(
                    RetrievedContext(
                        chunk=self.chunks[idx],
                        score=rrf_scores[idx],
                        retrieval_method="rrf"
                    )
                )
                
            return results

    def save(self, directory: str) -> None:
        """Serialize the store to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            # Save FAISS index
            faiss.write_index(self.faiss_index, str(path / "index.faiss"))
            
            # Save chunk registry and tokens
            state = {
                "chunks": self.chunks,
                "tokenized_corpus": self._tokenized_corpus
            }
            with open(path / "registry.pkl", "wb") as f:
                pickle.dump(state, f)

    def load(self, directory: str) -> None:
        """Load the store from disk."""
        path = Path(directory)

        with self._lock:
            # Reset internal state first. Call _init_empty_state() directly
            # rather than clear() to avoid constructing a FAISS index that
            # would be immediately discarded by faiss.read_index() below.
            self._init_empty_state()

            # Load FAISS
            self.faiss_index = faiss.read_index(str(path / "index.faiss"))

            # Load registry
            with open(path / "registry.pkl", "rb") as f:
                state = pickle.load(f)

            self.chunks = state["chunks"]
            self._tokenized_corpus = state["tokenized_corpus"]

            if self._tokenized_corpus:
                self.bm25 = BM25Okapi(self._tokenized_corpus)
