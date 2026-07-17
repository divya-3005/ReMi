"""
tests/test_store.py
───────────────────
Phase 5 test suite: HybridStore.

Validates FAISS dense retrieval, BM25 sparse retrieval, and Reciprocal Rank Fusion.
Also tests thread-safety (concurrent read-during-write) and serialization.
All embeddings are mocked via `mock_embedder`.
"""

from __future__ import annotations

import os
import threading
from unittest.mock import MagicMock

import pytest

from src.models.schemas import Chunk, DocumentMetadata
from src.vectorstore.store import HybridStore


@pytest.fixture
def store(mock_embedder, mock_settings):
    """Return an empty HybridStore with a mocked embedder."""
    return HybridStore(mock_embedder, mock_settings)


def make_chunks(num: int, start_idx: int = 0) -> list[Chunk]:
    """Helper to generate dummy chunks."""
    return [
        Chunk(
            doc_id=f"doc-{i}",
            text=f"This is the text for chunk number {i}",
            page_number=1,
            char_start=0,
            char_end=10,
        )
        for i in range(start_idx, start_idx + num)
    ]


class TestHybridStore:
    def test_add_chunks_increases_count(self, store):
        assert store.count() == 0
        chunks = make_chunks(50)
        store.add_chunks(chunks)
        assert store.count() == 50

    def test_search_returns_k_results(self, store):
        chunks = make_chunks(20)
        store.add_chunks(chunks)
        results = store.search("test query", k=5)
        assert len(results) == 5

    def test_search_on_empty_store_returns_empty_list(self, store):
        results = store.search("test", k=5)
        assert results == []

    def test_bm25_ranking_influence(self, store):
        """
        Verify that a chunk containing exact keyword matches is ranked higher
        than one that doesn't, proving BM25 is contributing to the RRF score.
        """
        # mock_embedder returns zero vectors for everything, so FAISS dense
        # scores will be identical for all chunks (0.0).
        # This means the final RRF ranking will depend ENTIRELY on BM25 sparse scores.
        chunks = make_chunks(5)
        chunks[3].text = "The unique keyword we are searching for."
        store.add_chunks(chunks)
        
        results = store.search("unique keyword", k=5)
        
        # The chunk with the exact keyword should be ranked #1
        assert results[0].chunk.chunk_id == chunks[3].chunk_id
        # It should have a sparse (or rrf) contribution
        assert results[0].score > 0

    def test_save_and_load(self, store, tmp_path, mock_embedder, mock_settings):
        # 1. Add data to original store
        chunks = make_chunks(10)
        store.add_chunks(chunks)
        assert store.count() == 10
        
        # 2. Save to disk
        save_dir = tmp_path / "vectorstore"
        store.save(str(save_dir))
        
        assert (save_dir / "index.faiss").exists()
        assert (save_dir / "registry.pkl").exists()
        
        # 3. Load into new store
        new_store = HybridStore(mock_embedder, mock_settings)
        new_store.load(str(save_dir))
        
        assert new_store.count() == 10
        
        # 4. Verify search still works on loaded store
        results = new_store.search("chunk", k=2)
        assert len(results) == 2

    def test_clear_removes_all_data(self, store):
        chunks = make_chunks(10)
        store.add_chunks(chunks)
        assert store.count() == 10
        
        store.clear()
        assert store.count() == 0
        assert store.search("test", k=5) == []


class TestStoreConcurrency:
    def test_concurrent_add_chunks(self, store):
        """Verify that multiple threads adding chunks don't corrupt the index."""
        # Create 4 lists of 25 chunks each
        batches = [make_chunks(25, start_idx=i*25) for i in range(4)]
        
        def worker(batch):
            store.add_chunks(batch)
            
        threads = [threading.Thread(target=worker, args=(b,)) for b in batches]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        assert store.count() == 100

    def test_read_during_write_is_safe(self, store):
        """
        Verify that a search called concurrently with add_chunks does not
        crash or return corrupt data. (Relies on the RLock in store.py)
        """
        store.add_chunks(make_chunks(10, start_idx=0))
        
        exceptions = []
        search_results = []
        
        def writer():
            try:
                # Add a large batch to increase the chance of collision
                store.add_chunks(make_chunks(100, start_idx=10))
            except Exception as e:
                exceptions.append(e)
                
        def reader():
            try:
                # Search repeatedly while the writer is running
                for _ in range(50):
                    res = store.search("chunk", k=5)
                    search_results.append(len(res))
            except Exception as e:
                exceptions.append(e)

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)
        
        t_write.start()
        t_read.start()
        
        t_write.join()
        t_read.join()
        
        assert not exceptions, f"Concurrency errors occurred: {exceptions}"
        assert store.count() == 110
        assert all(count >= 5 for count in search_results), "A search returned too few results"
