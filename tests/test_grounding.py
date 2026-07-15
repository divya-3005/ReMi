import pytest
from unittest.mock import patch
from models.research import Finding, LinkedAnswer, LinkedSentence, EvidenceSpan
from grounding.extractor import extract_spans
from grounding.linker import link
from grounding.scorer import faithfulness_score, coverage_score
from grounding.report import render
from vectorstore.retriever import SearchResult

def test_extract_spans_mocked():
    # Test simple extraction logic mocking the embeddings
    chunks = [SearchResult("c1", "d1", "file.txt", 1.0, "This is sentence one. This is sentence two.", 1)]
    
    with patch("grounding.extractor.embed_texts") as mock_embed:
        import numpy as np
        # Answer has 1 sentence. 
        # Chunks has 2 sentences. With window=3, there's just 1 window: "This is sentence one. This is sentence two."
        mock_embed.side_effect = [
            np.array([[1.0, 0.0]]), # answer embeddings
            np.array([[1.0, 0.0]])  # window embeddings
        ]
        
        spans = extract_spans("This is sentence one.", chunks, window_size=3)
        assert len(spans) == 1
        assert spans[0].chunk_index == 1
        assert spans[0].source_file == "file.txt"
        assert spans[0].relevance_score > 0.9 # should be 1.0

def test_linker():
    chunks = [SearchResult("c1", "d1", "file.txt", 1.0, "This is sentence one. This is sentence two.", 1)]
    finding = Finding("sq1", "This is sentence one.", chunks, 1.0)
    
    with patch("grounding.linker.extract_spans") as mock_extract:
        mock_extract.return_value = [
            EvidenceSpan("c1", "file.txt", 0, 20, "This is sentence one.", 0.9)
        ]
        
        linked = link("This is sentence one.", [finding])
        assert len(linked.sentences) == 1
        assert linked.sentences[0].grounded is True

def test_scorer_faithfulness():
    linked = LinkedAnswer("A. B.", [
        LinkedSentence("A.", [], True),
        LinkedSentence("B.", [], False)
    ])
    assert faithfulness_score(linked) == 0.5

def test_scorer_coverage():
    linked = LinkedAnswer("A.", [
        LinkedSentence("A.", [EvidenceSpan("c1", "f", 0, 1, "A", 0.9)], True)
    ])
    
    with patch("grounding.scorer.embed_query") as mock_eq:
        with patch("grounding.scorer.embed_texts") as mock_et:
            import numpy as np
            mock_eq.return_value = np.array([[1.0, 0.0]])
            mock_et.return_value = np.array([[1.0, 0.0]])
            
            score = coverage_score("Question?", linked)
            assert score > 0.9 # exactly 1.0

def test_report_render():
    linked = LinkedAnswer("A. B.", [
        LinkedSentence("A.", [EvidenceSpan("c1", "file.txt", 0, 1, "A", 0.9)], True),
        LinkedSentence("B.", [], False)
    ])
    
    md = render(linked)
    assert "A. [^1]" in md
    assert "B." in md
    assert "[UNGROUNDED]" not in md
    assert "Section c1" in md
    assert "**Sources**" in md
    assert "[^1]: file.txt, Section c1" in md
