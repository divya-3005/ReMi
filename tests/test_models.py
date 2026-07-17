"""
tests/test_models.py
────────────────────
Phase 1 test suite: Pydantic data models.

Tests cover: valid construction, JSON roundtrip, validation rejection,
auto-generated IDs, and cross-model relationships.
NO external calls. Pure in-process logic only.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    AgentReport,
    Chunk,
    CitationLink,
    DocumentMetadata,
    EvaluationResult,
    ResearchPlan,
    ResearchResult,
    RetrievedContext,
    SubQuestion,
    WorkflowAttempt,
)


# ── DocumentMetadata ──────────────────────────────────────────────────────────

class TestDocumentMetadata:
    def test_valid_construction(self):
        meta = DocumentMetadata(filename="report.pdf", file_type="pdf", num_pages=10)
        assert meta.filename == "report.pdf"
        assert meta.file_type == "pdf"
        assert meta.num_pages == 10

    def test_doc_id_auto_generated(self):
        meta = DocumentMetadata(filename="a.txt", file_type="txt", num_pages=1)
        assert meta.doc_id is not None
        assert len(meta.doc_id) == 36  # UUID4 string length

    def test_two_instances_have_different_doc_ids(self):
        m1 = DocumentMetadata(filename="a.txt", file_type="txt", num_pages=1)
        m2 = DocumentMetadata(filename="b.txt", file_type="txt", num_pages=1)
        assert m1.doc_id != m2.doc_id

    def test_uploaded_at_is_set_automatically(self):
        before = datetime.now(timezone.utc)
        meta = DocumentMetadata(filename="a.txt", file_type="txt", num_pages=1)
        after = datetime.now(timezone.utc)
        assert before <= meta.uploaded_at <= after

    def test_invalid_file_type_rejected(self):
        with pytest.raises(ValidationError):
            DocumentMetadata(filename="a.docx", file_type="docx", num_pages=1)

    def test_json_roundtrip(self):
        meta = DocumentMetadata(filename="paper.pdf", file_type="pdf", num_pages=5)
        json_str = meta.model_dump_json()
        restored = DocumentMetadata.model_validate_json(json_str)
        assert restored.doc_id == meta.doc_id
        assert restored.filename == meta.filename


# ── Chunk ─────────────────────────────────────────────────────────────────────

class TestChunk:
    def test_valid_construction(self):
        chunk = Chunk(
            doc_id="abc-123",
            text="Hello world",
            page_number=1,
            char_start=0,
            char_end=11,
        )
        assert chunk.text == "Hello world"
        assert chunk.char_start == 0
        assert chunk.char_end == 11

    def test_chunk_id_auto_generated(self):
        chunk = Chunk(
            doc_id="abc", text="x", page_number=1, char_start=0, char_end=1
        )
        assert chunk.chunk_id is not None
        assert len(chunk.chunk_id) == 36

    def test_char_start_is_required(self):
        with pytest.raises(ValidationError):
            Chunk(doc_id="abc", text="x", page_number=1, char_end=1)

    def test_char_end_is_required(self):
        with pytest.raises(ValidationError):
            Chunk(doc_id="abc", text="x", page_number=1, char_start=0)

    def test_embedding_defaults_to_none(self):
        chunk = Chunk(
            doc_id="abc", text="x", page_number=1, char_start=0, char_end=1
        )
        assert chunk.embedding is None

    def test_json_roundtrip(self):
        chunk = Chunk(
            doc_id="doc-1", text="Test chunk", page_number=2,
            char_start=100, char_end=110
        )
        restored = Chunk.model_validate_json(chunk.model_dump_json())
        assert restored.chunk_id == chunk.chunk_id
        assert restored.char_start == 100


# ── SubQuestion ───────────────────────────────────────────────────────────────

class TestSubQuestion:
    def test_valid_construction(self):
        sq = SubQuestion(
            question="What is RAG?",
            search_queries=["RAG definition", "retrieval augmented generation"],
        )
        assert sq.question == "What is RAG?"
        assert len(sq.search_queries) == 2

    def test_empty_search_queries_rejected(self):
        with pytest.raises(ValidationError):
            SubQuestion(question="What?", search_queries=[])

    def test_json_roundtrip(self):
        sq = SubQuestion(question="Q?", search_queries=["q1", "q2"])
        restored = SubQuestion.model_validate_json(sq.model_dump_json())
        assert restored.question == "Q?"
        assert restored.search_queries == ["q1", "q2"]


# ── ResearchPlan ──────────────────────────────────────────────────────────────

class TestResearchPlan:
    def test_valid_construction(self):
        plan = ResearchPlan(
            original_query="Explain transformers",
            reasoning="Breaking into sub-topics",
            sub_questions=[
                SubQuestion(question="What is attention?", search_queries=["attention mechanism"]),
                SubQuestion(question="What is BERT?", search_queries=["BERT model"]),
            ],
        )
        assert len(plan.sub_questions) == 2
        assert plan.original_query == "Explain transformers"

    def test_empty_sub_questions_rejected(self):
        with pytest.raises(ValidationError):
            ResearchPlan(
                original_query="Q", reasoning="R", sub_questions=[]
            )


# ── EvaluationResult ─────────────────────────────────────────────────────────

class TestEvaluationResult:
    def test_valid_all_scores_at_boundaries(self):
        ev = EvaluationResult(
            citation_coverage=0.0,
            citation_utilization=1.0,
            answer_relevance=0.5,
            hallucination_risk=1.0,
        )
        assert ev.citation_coverage == 0.0

    def test_citation_coverage_above_one_rejected(self):
        with pytest.raises(ValidationError):
            EvaluationResult(
                citation_coverage=1.1,
                citation_utilization=0.5,
                answer_relevance=0.5,
                hallucination_risk=0.0,
            )

    def test_negative_score_rejected(self):
        with pytest.raises(ValidationError):
            EvaluationResult(
                citation_coverage=0.5,
                citation_utilization=-0.1,
                answer_relevance=0.5,
                hallucination_risk=0.5,
            )

    def test_hallucination_risk_consistency(self):
        """hallucination_risk should equal 1.0 - citation_coverage by convention."""
        ev = EvaluationResult(
            citation_coverage=0.7,
            citation_utilization=0.6,
            answer_relevance=0.8,
            hallucination_risk=0.3,
        )
        assert abs(ev.hallucination_risk - (1.0 - ev.citation_coverage)) < 1e-9

    def test_json_roundtrip(self):
        ev = EvaluationResult(
            citation_coverage=0.8, citation_utilization=0.6,
            answer_relevance=0.9, hallucination_risk=0.2,
        )
        restored = EvaluationResult.model_validate_json(ev.model_dump_json())
        assert restored.citation_coverage == 0.8


# ── CitationLink ──────────────────────────────────────────────────────────────

class TestCitationLink:
    def test_valid_construction(self):
        cl = CitationLink(
            footnote_id=1,
            chunk_id="chunk-abc",
            char_start=50,
            char_end=170,
            excerpt="This is the source text excerpt that was matched.",
        )
        assert cl.footnote_id == 1
        assert len(cl.excerpt) > 0

    def test_excerpt_truncated_to_120_chars(self):
        long_text = "x" * 200
        cl = CitationLink(
            footnote_id=1, chunk_id="c", char_start=0, char_end=200, excerpt=long_text
        )
        assert len(cl.excerpt) <= 120


# ── WorkflowAttempt ───────────────────────────────────────────────────────────

class TestWorkflowAttempt:
    def test_valid_no_retry(self):
        ev = EvaluationResult(
            citation_coverage=0.8, citation_utilization=0.7,
            answer_relevance=0.9, hallucination_risk=0.2,
        )
        attempt = WorkflowAttempt(
            attempt_number=1,
            evaluation=ev,
            triggered_retry=False,
            retry_reason=None,
        )
        assert attempt.attempt_number == 1
        assert not attempt.triggered_retry

    def test_retry_with_reason(self):
        ev = EvaluationResult(
            citation_coverage=0.3, citation_utilization=0.2,
            answer_relevance=0.5, hallucination_risk=0.7,
        )
        attempt = WorkflowAttempt(
            attempt_number=1,
            evaluation=ev,
            triggered_retry=True,
            retry_reason="citation_coverage=0.30, citation_utilization=0.20",
        )
        assert attempt.triggered_retry
        assert "citation_coverage" in attempt.retry_reason


# ── AgentReport ───────────────────────────────────────────────────────────────

class TestAgentReport:
    def _make_report(self) -> AgentReport:
        ev = EvaluationResult(
            citation_coverage=0.75, citation_utilization=0.60,
            answer_relevance=0.85, hallucination_risk=0.25,
        )
        attempt = WorkflowAttempt(
            attempt_number=1, evaluation=ev,
            triggered_retry=False, retry_reason=None,
        )
        cl = CitationLink(
            footnote_id=1, chunk_id="c-1",
            char_start=0, char_end=50, excerpt="Source text here.",
        )
        return AgentReport(
            query="What is RAG?",
            answer_text="# Answer\n\nRAG stands for... [^1]",
            citations=[cl],
            evaluation=ev,
            workflow_attempts=[attempt],
            elapsed_seconds=12.5,
            known_limitations_applied=["difflib_grounding"],
        )

    def test_valid_construction(self):
        report = self._make_report()
        assert report.query == "What is RAG?"
        assert report.elapsed_seconds > 0

    def test_elapsed_seconds_must_be_positive(self):
        with pytest.raises(ValidationError):
            ev = EvaluationResult(
                citation_coverage=0.5, citation_utilization=0.5,
                answer_relevance=0.5, hallucination_risk=0.5,
            )
            AgentReport(
                query="Q", answer_text="A", citations=[], evaluation=ev,
                workflow_attempts=[], elapsed_seconds=-1.0,
                known_limitations_applied=[],
            )

    def test_json_roundtrip(self):
        report = self._make_report()
        restored = AgentReport.model_validate_json(report.model_dump_json())
        assert restored.query == report.query
        assert len(restored.citations) == 1
        assert restored.evaluation.citation_coverage == 0.75
