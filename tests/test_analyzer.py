"""
tests/test_analyzer.py
──────────────────────
Phase 9 test suite: AnalyzerAgent.

Tests relevance scoring, threshold filtering, and the all-below-floor
edge case (keep the single best chunk, set low_confidence=True).
"""

from __future__ import annotations

from pydantic import BaseModel
from unittest.mock import MagicMock

import pytest

from src.agent.analyzer import AnalyzerAgent
from src.models.schemas import Chunk, ResearchResult, RetrievedContext, SubQuestion


def _make_chunk(chunk_id: str, text: str = "chunk text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id="d1", text=text,
        page_number=1, char_start=0, char_end=len(text)
    )


def _make_context(chunk_id: str, score: float = 0.8) -> RetrievedContext:
    return RetrievedContext(
        chunk=_make_chunk(chunk_id),
        score=score,
        retrieval_method="rrf",
    )


def _make_result(contexts: list[RetrievedContext]) -> ResearchResult:
    return ResearchResult(
        sub_question=SubQuestion(question="Q?", search_queries=["q"]),
        contexts=contexts,
        answer="Some answer.",
    )


@pytest.fixture
def mock_groq_client():
    client = MagicMock()
    # Default: return scores of 0.9 for everything
    return client


@pytest.fixture
def analyzer(mock_groq_client, mock_settings):
    return AnalyzerAgent(mock_groq_client, mock_settings)


def _set_scores(mock_groq_client, scores: list[float]):
    """Configure the mock to return a specific set of scores."""
    class ScoreResponse(BaseModel):
        scores: list[float]

    mock_groq_client.generate.return_value = ScoreResponse(scores=scores)


class TestAnalyzerAgent:
    def test_returns_list_of_research_results(self, analyzer, mock_groq_client):
        _set_scores(mock_groq_client, [0.9, 0.8])
        results = [_make_result([_make_context("c1"), _make_context("c2")])]
        output = analyzer.analyze(results, relevance_floor=0.40)
        assert isinstance(output, list)
        assert len(output) == 1

    def test_low_scoring_chunks_filtered_out(self, analyzer, mock_groq_client):
        _set_scores(mock_groq_client, [0.9, 0.1, 0.8])
        contexts = [_make_context("c1"), _make_context("c2"), _make_context("c3")]
        results = [_make_result(contexts)]
        output = analyzer.analyze(results, relevance_floor=0.40)
        remaining_ids = [ctx.chunk.chunk_id for ctx in output[0].contexts]
        assert "c1" in remaining_ids
        assert "c3" in remaining_ids
        assert "c2" not in remaining_ids

    def test_high_floor_filters_most_chunks(self, analyzer, mock_groq_client):
        _set_scores(mock_groq_client, [0.5, 0.3, 0.2])
        contexts = [_make_context("c1"), _make_context("c2"), _make_context("c3")]
        results = [_make_result(contexts)]
        output = analyzer.analyze(results, relevance_floor=0.80)
        # All below floor → should keep exactly 1 with low_confidence=True
        assert len(output[0].contexts) == 1
        assert output[0].low_confidence is True

    def test_all_below_floor_keeps_single_best(self, analyzer, mock_groq_client):
        _set_scores(mock_groq_client, [0.3, 0.5, 0.1])  # best is index 1 (0.5)
        contexts = [_make_context("c1"), _make_context("c2"), _make_context("c3")]
        results = [_make_result(contexts)]
        output = analyzer.analyze(results, relevance_floor=0.80)
        assert len(output[0].contexts) == 1
        assert output[0].contexts[0].chunk.chunk_id == "c2"  # highest score

    def test_all_above_floor_low_confidence_false(self, analyzer, mock_groq_client):
        _set_scores(mock_groq_client, [0.9, 0.85])
        contexts = [_make_context("c1"), _make_context("c2")]
        results = [_make_result(contexts)]
        output = analyzer.analyze(results, relevance_floor=0.40)
        assert output[0].low_confidence is False

    def test_empty_contexts_returns_empty_contexts(self, analyzer, mock_groq_client):
        """Result with no contexts should pass through without LLM call."""
        results = [_make_result([])]
        output = analyzer.analyze(results, relevance_floor=0.40)
        assert output[0].contexts == []
        mock_groq_client.generate.assert_not_called()

    def test_multiple_results_each_scored_independently(self, analyzer, mock_groq_client):
        """Each ResearchResult's contexts are scored separately."""
        _set_scores(mock_groq_client, [0.9])  # called once per result
        r1 = _make_result([_make_context("c1")])
        r2 = _make_result([_make_context("c2")])
        output = analyzer.analyze([r1, r2], relevance_floor=0.40)
        assert len(output) == 2
        assert mock_groq_client.generate.call_count == 2
