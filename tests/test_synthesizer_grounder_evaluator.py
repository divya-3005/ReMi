"""
tests/test_synthesizer_grounder_evaluator.py
─────────────────────────────────────────────
Phases 10, 11, 12: Synthesizer, Grounder, and Evaluator tests.

All LLM/embedding calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent.synthesizer import SynthesizerAgent
from src.evaluation.evaluator import EvaluatorAgent
from src.grounding.grounder import GrounderAgent
from src.models.schemas import (
    Chunk, EvaluationResult, ResearchResult, RetrievedContext, SubQuestion
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_chunk(chunk_id: str = "c1", text: str = "The Federal Reserve raised rates.") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id="d1", text=text,
        page_number=1, char_start=0, char_end=len(text)
    )


def _make_context(chunk: Chunk) -> RetrievedContext:
    return RetrievedContext(chunk=chunk, score=0.9, retrieval_method="rrf")


def _make_result(answer: str, low_confidence: bool = False) -> ResearchResult:
    return ResearchResult(
        sub_question=SubQuestion(question="Q?", search_queries=["q"]),
        contexts=[_make_context(_make_chunk())],
        answer=answer,
        low_confidence=low_confidence,
    )


# ── SynthesizerAgent ──────────────────────────────────────────────────────────

class TestSynthesizerAgent:
    @pytest.fixture
    def mock_gemini_client(self):
        client = MagicMock()
        client.generate_text.return_value = "# Research Report\n\nThe answer is clear. [^1]"
        return client

    @pytest.fixture
    def synthesizer(self, mock_gemini_client):
        return SynthesizerAgent(mock_gemini_client)

    def test_returns_string_report(self, synthesizer):
        results = [_make_result("Answer A."), _make_result("Answer B.")]
        report = synthesizer.synthesize("What is X?", results)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_low_confidence_sections_flagged_in_prompt(self, synthesizer, mock_gemini_client):
        results = [_make_result("Answer A.", low_confidence=True)]
        synthesizer.synthesize("Q?", results)
        prompt_used = mock_gemini_client.generate_text.call_args.args[0]
        assert "⚠️" in prompt_used or "LOW CONFIDENCE" in prompt_used

    def test_normal_result_not_flagged(self, synthesizer, mock_gemini_client):
        results = [_make_result("Normal answer.", low_confidence=False)]
        synthesizer.synthesize("Q?", results)
        prompt_used = mock_gemini_client.generate_text.call_args.args[0]
        assert "LOW CONFIDENCE" not in prompt_used

    def test_all_answers_included_in_prompt(self, synthesizer, mock_gemini_client):
        results = [
            _make_result("Answer alpha."),
            _make_result("Answer beta."),
        ]
        synthesizer.synthesize("Q?", results)
        prompt_used = mock_gemini_client.generate_text.call_args.args[0]
        assert "Answer alpha." in prompt_used
        assert "Answer beta." in prompt_used


# ── GrounderAgent ─────────────────────────────────────────────────────────────

class TestGrounderAgent:
    @pytest.fixture
    def grounder(self):
        return GrounderAgent()

    def test_no_citations_on_empty_report(self, grounder):
        _, citations = grounder.ground("Report with no footnotes.", [_make_chunk()])
        assert citations == []

    def test_report_returned_unchanged(self, grounder):
        report = "Some report text with [^1] a citation."
        chunk = _make_chunk(text="Some report text with a citation")
        returned_report, _ = grounder.ground(report, [chunk])
        assert returned_report == report

    def test_no_chunks_returns_empty_citations(self, grounder):
        _, citations = grounder.ground("Report [^1] with citation.", [])
        assert citations == []

    def test_matching_chunk_produces_citation_link(self, grounder):
        # The claim in the report should match the chunk text closely,
        # but the chunk itself is much longer (which previously broke difflib ratio).
        long_text = "The Federal Reserve raised interest rates significantly in 2022. " + "This is extra padding. " * 50
        chunk = _make_chunk(
            chunk_id="c1",
            text=long_text
        )
        report = "The Federal Reserve raised interest rates significantly in 2022. [^1]"
        _, citations = grounder.ground(report, [chunk])
        # Should find at least one citation
        assert len(citations) >= 1
        assert citations[0].footnote_id == 1

    def test_very_dissimilar_chunk_produces_no_citation(self, grounder):
        # Completely unrelated chunk
        chunk = _make_chunk(text="The history of ancient Roman architecture.")
        report = "Quantum computing uses qubits. [^1]"
        _, citations = grounder.ground(report, [chunk])
        assert citations == []

    def test_duplicate_footnotes_linked_once(self, grounder):
        chunk = _make_chunk(text="The Federal Reserve raised rates.")
        # [^1] appears twice in the report
        report = "The Federal Reserve raised rates. [^1] Also, rates were raised. [^1]"
        _, citations = grounder.ground(report, [chunk])
        footnote_ids = [c.footnote_id for c in citations]
        assert footnote_ids.count(1) == 1  # Only linked once


# ── EvaluatorAgent ────────────────────────────────────────────────────────────

class TestEvaluatorAgent:
    @pytest.fixture
    def mock_embedder(self, mock_settings):
        embedder = MagicMock()
        dim = mock_settings.embedding_dim
        embedder.embed_query.return_value = [1.0] + [0.0] * (dim - 1)
        embedder.embed_texts.return_value = [[1.0] + [0.0] * (dim - 1)]
        return embedder

    @pytest.fixture
    def evaluator(self, mock_embedder):
        return EvaluatorAgent(mock_embedder)

    def test_returns_evaluation_result(self, evaluator):
        chunk = _make_chunk()
        ev = evaluator.evaluate("Q?", "Report [^1].", [_make_context(chunk)], [chunk])
        assert isinstance(ev, EvaluationResult)

    def test_all_scores_in_range(self, evaluator):
        chunk = _make_chunk()
        ev = evaluator.evaluate("Q?", "Report [^1].", [_make_context(chunk)], [chunk])
        for score in [ev.citation_coverage, ev.citation_utilization,
                      ev.answer_relevance, ev.hallucination_risk]:
            assert 0.0 <= score <= 1.0

    def test_hallucination_risk_is_one_minus_coverage(self, evaluator):
        chunk = _make_chunk()
        ev = evaluator.evaluate("Q?", "Report [^1].", [_make_context(chunk)], [chunk])
        assert abs(ev.hallucination_risk - (1.0 - ev.citation_coverage)) < 1e-4

    def test_no_footnotes_in_report_gives_zero_coverage(self, evaluator):
        chunk = _make_chunk()
        ev = evaluator.evaluate("Q?", "Report with no citations.", [_make_context(chunk)], [chunk])
        assert ev.citation_coverage == 0.0
        assert ev.hallucination_risk == 1.0

    def test_no_contexts_gives_zero_utilization(self, evaluator):
        ev = evaluator.evaluate("Q?", "Report [^1].", [], [])
        assert ev.citation_utilization == 0.0

    def test_answer_relevance_is_one_for_identical_embeddings(self, evaluator, mock_embedder, mock_settings):
        """When query and report embeddings are identical, cosine sim = 1.0."""
        dim = mock_settings.embedding_dim
        identical_vector = [1.0 / dim**0.5] * dim
        mock_embedder.embed_query.return_value = identical_vector
        mock_embedder.embed_texts.return_value = [identical_vector]
        chunk = _make_chunk()
        ev = evaluator.evaluate("Q?", "Report.", [_make_context(chunk)], [])
        assert abs(ev.answer_relevance - 1.0) < 1e-4

    def test_embedding_failure_defaults_to_zero_relevance(self, evaluator, mock_embedder):
        """If the embedder raises, answer_relevance should be 0.0, not a crash."""
        mock_embedder.embed_query.side_effect = RuntimeError("API down")
        chunk = _make_chunk()
        ev = evaluator.evaluate("Q?", "Report.", [_make_context(chunk)], [])
        assert ev.answer_relevance == 0.0
