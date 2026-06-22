import pytest
from unittest.mock import patch, MagicMock
from genai.qa import answer, QAResult
from genai.summarizer import summarize_document, DocumentSummary
from vectorstore.retriever import SearchResult
from storage.document_store import DocumentStore
from models.document import Document, Chunk

@patch("genai.qa.complete")
@patch("genai.qa.search")
def test_answer(mock_search, mock_complete):
    mock_search.return_value = [
        SearchResult("c1", "doc1", "file.txt", 0.9, "Some text about AI.", 1)
    ]
    mock_complete.return_value = "AI is cool. [source: file.txt, chunk 1]"
    
    # Store is not used in mock, pass None
    result = answer("What is AI?", None)
    
    assert result.query == "What is AI?"
    assert result.answer == "AI is cool. [source: file.txt, chunk 1]"
    assert len(result.sources) == 1
    assert "Some text about AI." in result.context_used
    assert "[source: file.txt, chunk 1]" in result.context_used

@patch("genai.summarizer.complete")
def test_summarize_document_zero_chunks(mock_complete):
    mock_store = MagicMock(spec=DocumentStore)
    doc = Document("doc1", "file.txt", "", 0, "now")
    mock_store.get_by_id.return_value = (doc, [])
    
    result = summarize_document("doc1", mock_store)
    
    assert result.full_summary == "Document has no text chunks to summarize."
    assert len(result.chunk_summaries) == 0
    mock_complete.assert_not_called()

@patch("genai.summarizer.complete")
def test_summarize_document(mock_complete):
    mock_store = MagicMock(spec=DocumentStore)
    doc = Document("doc1", "file.txt", "", 0, "now")
    doc.nlp = {
        "keywords": {"document_level": [{"keyword": "AI", "score": 0.5}]},
        "entities": {"document_level": [{"text": "Google", "label": "ORG", "count": 1}]}
    }
    # Using kwargs for Chunk constructor correctly based on our previous fix
    chunks = [
        Chunk(chunk_id="c1", document_id="doc1", content="text 1", source_file="file.txt", page_number=1, char_offset=0, token_count=2)
    ]
    mock_store.get_by_id.return_value = (doc, chunks)
    
    # complete is called twice: once for chunk, once for doc summary
    mock_complete.side_effect = ["Chunk summary", "Full summary"]
    
    result = summarize_document("doc1", mock_store)
    
    assert result.doc_id == "doc1"
    assert result.full_summary == "Full summary"
    assert "Chunk summary" in result.chunk_summaries
    assert "AI" in result.top_keywords
    assert "Google (ORG)" in result.top_entities
    assert mock_complete.call_count == 2
