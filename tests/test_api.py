import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("api.routes.documents.load_file")
@patch("api.routes.documents.clean_text")
@patch("api.routes.documents.chunk_document")
def test_ingest_document(mock_chunk, mock_clean, mock_load):
    mock_load.return_value = ("Test content", {"file_size_bytes": 100, "page_count": 1, "ingested_at": "now"})
    mock_clean.return_value = "Test content"
    mock_chunk.return_value = []
    
    app.state.store = MagicMock()
    app.state.vstore = MagicMock()
    
    response = client.post(
        "/api/documents/ingest",
        files={"file": ("test.txt", b"Test content")}
    )
    
    assert response.status_code == 200
    assert "doc_id" in response.json()
    assert response.json()["filename"] == "test.txt"

@patch("api.routes.search.vector_search")
def test_search(mock_search):
    mock_search.return_value = [
        MagicMock(chunk_index=1, doc_id="d1", source_file="test.txt", score=0.9, chunk_text="Test")
    ]
    
    app.state.vstore = MagicMock()
    
    response = client.post(
        "/api/search",
        json={"query": "test"}
    )
    
    assert response.status_code == 200
    res = response.json()
    assert len(res["results"]) == 1
    assert res["results"][0]["score"] == 0.9

@patch("api.routes.qa.genai_ask")
@patch("api.routes.qa.link")
@patch("api.routes.qa.faithfulness_score")
@patch("api.routes.qa.coverage_score")
@patch("api.routes.qa.render")
def test_qa(mock_render, mock_cov, mock_faith, mock_link, mock_ask):
    mock_ask.return_value = MagicMock(answer="Answer", sources=[])
    mock_link.return_value = MagicMock()
    mock_render.return_value = "Grounded Answer"
    mock_faith.return_value = 1.0
    mock_cov.return_value = 1.0
    
    app.state.vstore = MagicMock()
    
    response = client.post(
        "/api/qa",
        json={"query": "test"}
    )
    
    assert response.status_code == 200
    res = response.json()
    assert res["answer"] == "Grounded Answer"
    assert res["faithfulness_score"] == 1.0

def test_list_reports():
    with patch("api.routes.research.os.path.exists") as mock_exists:
        with patch("api.routes.research.glob.glob") as mock_glob:
            mock_exists.return_value = True
            mock_glob.return_value = []
            
            response = client.get("/api/research/reports")
            assert response.status_code == 200
            assert response.json() == []
