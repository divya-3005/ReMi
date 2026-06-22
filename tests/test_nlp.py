import pytest
from nlp.keywords import extract_keywords
from nlp.ner import extract_entities
from nlp.importance import score_sentences

def test_extract_keywords():
    chunks = [
        {"chunk_id": "c1", "content": "The artificial intelligence system uses neural networks."},
        {"chunk_id": "c2", "content": "Neural networks are models of artificial intelligence."}
    ]
    res = extract_keywords(chunks, top_n=2)
    assert "per_chunk" in res
    assert "document_level" in res
    assert len(res["document_level"]) <= 2
    assert len(res["per_chunk"]["c1"]) <= 2

def test_extract_entities():
    text = "Apple Inc. announced new products in San Francisco on Tuesday."
    ents = extract_entities(text)
    labels = {e["label"] for e in ents}
    
    # Check if ORG or PERSON or GPE was found
    assert "ORG" in labels or "PERSON" in labels or "GPE" in labels
    
    # Specific check for Apple
    apple_ent = next((e for e in ents if e["text"] == "Apple Inc."), None)
    assert apple_ent is not None
    assert apple_ent["label"] == "ORG"
    assert apple_ent["count"] == 1

def test_score_sentences():
    text = "Machine learning is fascinating. It allows computers to learn from data without explicit programming. Neural networks are a subset of machine learning."
    sents = score_sentences(text, top_n=2)
    
    assert len(sents) == 2
    total_score = sum(s["score"] for s in sents)
    assert total_score > 0
