import os
import shutil
import pytest
from vectorstore.store import FaissStore
from vectorstore.retriever import search
from models.document import Chunk

@pytest.fixture
def temp_store(tmp_path):
    data_dir = str(tmp_path / "data")
    store = FaissStore(data_dir=data_dir)
    yield store

@pytest.fixture(autouse=True)
def mock_embedder(monkeypatch):
    import numpy as np
    def mock_embed_texts(texts):
        out = np.zeros((len(texts), 3072), dtype=np.float32)
        for i, t in enumerate(texts):
            if "dogs" in t.lower() or "puppies" in t.lower():
                out[i, 0] = 1.0
            elif "weather" in t.lower() or "sunny" in t.lower():
                out[i, 1] = 1.0
            else:
                out[i, 2] = 1.0
        return out

    def mock_embed_query(query):
        out = np.zeros((1, 3072), dtype=np.float32)
        if "dogs" in query.lower() or "puppies" in query.lower():
            out[0, 0] = 1.0
        elif "weather" in query.lower() or "sunny" in query.lower():
            out[0, 1] = 1.0
        else:
            out[0, 2] = 1.0
        return out

    def mock_local_embed_texts(texts):
        return np.zeros((len(texts), 384), dtype=np.float32)

    def mock_local_embed_query(query):
        return np.zeros((1, 384), dtype=np.float32)

    monkeypatch.setattr("vectorstore.store.embed_texts", mock_embed_texts)
    monkeypatch.setattr("vectorstore.store.local_embed_texts", mock_local_embed_texts)
    monkeypatch.setattr("vectorstore.retriever.embed_query", mock_embed_query)
    monkeypatch.setattr("vectorstore.retriever.local_embed_query", mock_local_embed_query)

def test_embed_and_search(temp_store):
    doc_id = "test-doc-1"
    chunks = [
        Chunk(chunk_id="c1", document_id=doc_id, content="Artificial intelligence is fascinating.", source_file="test.txt", page_number=1, char_offset=0, token_count=5),
        Chunk(chunk_id="c2", document_id=doc_id, content="Dogs are great pets.", source_file="test.txt", page_number=1, char_offset=0, token_count=5),
        Chunk(chunk_id="c3", document_id=doc_id, content="The weather is sunny today.", source_file="test.txt", page_number=1, char_offset=0, token_count=5)
    ]
    
    temp_store.add_document(doc_id, chunks, source_file="test.txt")
    
    assert temp_store.total_chunks() == 3
    
    # Search
    results = search("I love puppies", temp_store, top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "c2"
    assert 0.0 <= results[0].score <= 1.0001  # FP math can yield slightly > 1.0

def test_remove_document(temp_store):
    doc_id_1 = "doc1"
    doc_id_2 = "doc2"
    
    c1 = Chunk(chunk_id="c1", document_id=doc_id_1, content="Doc 1 content", source_file="test.txt", page_number=1, char_offset=0, token_count=3)
    c2 = Chunk(chunk_id="c2", document_id=doc_id_2, content="Doc 2 content", source_file="test.txt", page_number=1, char_offset=0, token_count=3)
    
    temp_store.add_document(doc_id_1, [c1])
    temp_store.add_document(doc_id_2, [c2])
    
    assert temp_store.total_chunks() == 2
    
    temp_store.remove_document(doc_id_1)
    
    assert temp_store.total_chunks() == 1
    assert temp_store.metadata[0]["doc_id"] == doc_id_2
