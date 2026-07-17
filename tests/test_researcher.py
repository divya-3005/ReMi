"""
tests/test_researcher.py
────────────────────────
Phase 8 test suite: ResearcherAgent.

Tests HyDE generation, multi-query retrieval, deduplication, and
the zero-results graceful fallback.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent.researcher import ResearcherAgent
from src.models.schemas import Chunk, RetrievedContext, ResearchResult, SubQuestion


def _make_sub_question(queries: list[str] | None = None) -> SubQuestion:
    return SubQuestion(
        question="What caused the subprime mortgage crisis?",
        search_queries=queries or ["subprime mortgage crisis causes", "housing bubble 2008"]
    )


def _make_chunk(chunk_id: str = "c1", text: str = "Some document text.") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        text=text,
        page_number=1,
        char_start=0,
        char_end=len(text),
    )


def _make_retrieved_context(chunk: Chunk, score: float = 0.9) -> RetrievedContext:
    return RetrievedContext(chunk=chunk, score=score, retrieval_method="rrf")


@pytest.fixture
def mock_store():
    store = MagicMock()
    chunk = _make_chunk("c1", "Subprime loans were risky instruments.")
    store.search.return_value = [_make_retrieved_context(chunk)]
    return store


@pytest.fixture
def mock_groq_client():
    client = MagicMock()
    client.generate_text.return_value = "Subprime loans were issued to borrowers with poor credit."
    return client


@pytest.fixture
def researcher(mock_store, mock_groq_client, mock_settings):
    return ResearcherAgent(mock_store, mock_groq_client, mock_settings)


class TestResearcherAgent:
    def test_returns_research_result(self, researcher):
        sq = _make_sub_question()
        result = researcher.research(sq)
        assert isinstance(result, ResearchResult)

    def test_result_sub_question_matches_input(self, researcher):
        sq = _make_sub_question()
        result = researcher.research(sq)
        assert result.sub_question.question == sq.question

    def test_result_has_non_empty_answer(self, researcher):
        sq = _make_sub_question()
        result = researcher.research(sq)
        assert len(result.answer) > 0

    def test_store_searched_for_each_query_variant(self, researcher, mock_store):
        sq = _make_sub_question(["query A", "query B", "query C"])
        researcher.research(sq)
        # Should search once per search_query in sub_question
        assert mock_store.search.call_count == len(sq.search_queries)

    def test_duplicate_chunks_are_deduplicated(self, researcher, mock_store, mock_settings):
        """Two different search queries returning the same chunk_id → only one context."""
        chunk = _make_chunk("c-same")
        ctx = _make_retrieved_context(chunk)
        # Both searches return the same chunk
        mock_store.search.return_value = [ctx]

        sq = _make_sub_question(["query A", "query B"])
        result = researcher.research(sq)

        chunk_ids = [ctx.chunk.chunk_id for ctx in result.contexts]
        assert len(chunk_ids) == len(set(chunk_ids))  # no duplicates

    def test_k_defaults_to_settings_retrieval_k(self, researcher, mock_store, mock_settings):
        sq = _make_sub_question(["single query"])
        researcher.research(sq)
        call_kwargs = mock_store.search.call_args.kwargs
        assert call_kwargs.get("k") == mock_settings.retrieval_k

    def test_k_override_passed_through(self, researcher, mock_store):
        sq = _make_sub_question(["query"])
        researcher.research(sq, k=3)
        call_kwargs = mock_store.search.call_args.kwargs
        assert call_kwargs.get("k") == 3

    def test_zero_results_returns_graceful_result(self, researcher, mock_store):
        """When all searches return empty, should NOT crash — returns sentinel answer."""
        mock_store.search.return_value = []
        sq = _make_sub_question()
        result = researcher.research(sq)
        assert isinstance(result, ResearchResult)
        assert result.contexts == []
        assert "no relevant" in result.answer.lower() or "not found" in result.answer.lower()

    def test_zero_results_does_not_call_llm(self, researcher, mock_store, mock_groq_client):
        """Empty retrieval → no point calling the LLM for synthesis."""
        mock_store.search.return_value = []
        sq = _make_sub_question()
        researcher.research(sq)
        mock_groq_client.generate_text.assert_not_called()
