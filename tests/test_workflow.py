"""
tests/test_workflow.py
──────────────────────
Phase 13 test suite: ResearchWorkflow.

Tests the full agentic feedback loop including:
- Happy path (first attempt passes quality gates)
- Retry path (first attempt fails → reformulation → second attempt)
- Retry exhaustion (all attempts fail → returns best result)
- contexts/cited_chunks derivation
- WorkflowAttempt audit trail correctness
- elapsed_seconds populated
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, call, patch

import pytest

from src.agent.workflow import ResearchWorkflow
from src.models.schemas import (
    AgentReport,
    CitationLink,
    Chunk,
    EvaluationResult,
    ResearchPlan,
    ResearchResult,
    RetrievedContext,
    SubQuestion,
    WorkflowAttempt,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_sub_question(q: str = "Q?") -> SubQuestion:
    return SubQuestion(question=q, search_queries=["sq"])


def _make_plan(q: str = "Q?") -> ResearchPlan:
    return ResearchPlan(
        original_query=q,
        reasoning="reasoning",
        sub_questions=[_make_sub_question(q)],
    )


def _make_chunk(chunk_id: str = "c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id="d1", text="Source text.",
        page_number=1, char_start=0, char_end=12,
    )


def _make_context(chunk: Chunk) -> RetrievedContext:
    return RetrievedContext(chunk=chunk, score=0.9, retrieval_method="rrf")


def _make_research_result(chunk: Chunk) -> ResearchResult:
    return ResearchResult(
        sub_question=_make_sub_question(),
        contexts=[_make_context(chunk)],
        answer="Answer text.",
    )


def _make_eval(coverage: float = 0.8, utilization: float = 0.7) -> EvaluationResult:
    return EvaluationResult(
        citation_coverage=coverage,
        citation_utilization=utilization,
        answer_relevance=0.85,
        hallucination_risk=round(1.0 - coverage, 4),
    )


def _make_citation(chunk_id: str = "c1") -> CitationLink:
    return CitationLink(
        footnote_id=1, chunk_id=chunk_id,
        char_start=0, char_end=12, excerpt="Source text."
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_planner():
    p = MagicMock()
    p.plan.return_value = _make_plan()
    p.plan_from_prompt.return_value = _make_plan()
    return p


@pytest.fixture
def mock_researcher():
    chunk = _make_chunk()
    r = MagicMock()
    r.research.return_value = _make_research_result(chunk)
    return r


@pytest.fixture
def mock_analyzer():
    chunk = _make_chunk()
    a = MagicMock()
    a.analyze.return_value = [_make_research_result(chunk)]
    return a


@pytest.fixture
def mock_synthesizer():
    s = MagicMock()
    s.synthesize.return_value = "# Report\n\nSome text [^1]"
    return s


@pytest.fixture
def mock_grounder():
    g = MagicMock()
    chunk = _make_chunk()
    g.ground.return_value = ("# Report\n\nSome text [^1]", [_make_citation(chunk.chunk_id)])
    return g


@pytest.fixture
def mock_evaluator():
    e = MagicMock()
    e.evaluate.return_value = _make_eval(coverage=0.8, utilization=0.7)
    return e


@pytest.fixture
def mock_store():
    s = MagicMock()
    s.chunks = [_make_chunk()]  # non-empty store
    return s


@pytest.fixture
def workflow(
    mock_planner, mock_researcher, mock_analyzer,
    mock_synthesizer, mock_grounder, mock_evaluator,
    mock_store, mock_settings,
):
    return ResearchWorkflow(
        planner=mock_planner,
        researcher=mock_researcher,
        analyzer=mock_analyzer,
        synthesizer=mock_synthesizer,
        grounder=mock_grounder,
        evaluator=mock_evaluator,
        store=mock_store,
        settings=mock_settings,
    )


# ── Happy path ────────────────────────────────────────────────────────────────

class TestWorkflowHappyPath:
    def test_returns_agent_report(self, workflow):
        report = workflow.run("What is RAG?")
        assert isinstance(report, AgentReport)

    def test_query_preserved_in_report(self, workflow):
        report = workflow.run("What is RAG?")
        assert report.query == "What is RAG?"

    def test_single_attempt_when_scores_pass(self, workflow):
        report = workflow.run("What is RAG?")
        assert len(report.workflow_attempts) == 1
        assert report.workflow_attempts[0].triggered_retry is False

    def test_planner_plan_called_once_on_first_attempt(self, workflow, mock_planner):
        workflow.run("Query")
        mock_planner.plan.assert_called_once_with("Query")
        mock_planner.plan_from_prompt.assert_not_called()

    def test_analyzer_called_with_relevance_floor(self, workflow, mock_analyzer, mock_settings):
        workflow.run("Query")
        call_kwargs = mock_analyzer.analyze.call_args
        assert call_kwargs.kwargs.get("relevance_floor") == mock_settings.analyzer_relevance_floor or \
               call_kwargs.args[1] == mock_settings.analyzer_relevance_floor

    def test_elapsed_seconds_populated(self, workflow):
        report = workflow.run("Query")
        assert report.elapsed_seconds > 0

    def test_known_limitations_populated(self, workflow):
        report = workflow.run("Query")
        assert "difflib_grounding" in report.known_limitations_applied

    def test_citations_in_report(self, workflow):
        report = workflow.run("Query")
        assert len(report.citations) >= 1


# ── Retry path ────────────────────────────────────────────────────────────────

class TestWorkflowRetryPath:
    def test_retry_triggered_on_low_coverage(
        self, mock_planner, mock_researcher, mock_analyzer,
        mock_synthesizer, mock_grounder, mock_evaluator,
        mock_store, mock_settings,
    ):
        # First attempt: low scores → retry triggered
        # Second attempt: good scores → done
        mock_evaluator.evaluate.side_effect = [
            _make_eval(coverage=0.20, utilization=0.10),  # fail
            _make_eval(coverage=0.85, utilization=0.75),  # pass
        ]
        mock_settings.__dict__["max_workflow_retries"] = 1

        wf = ResearchWorkflow(
            planner=mock_planner, researcher=mock_researcher, analyzer=mock_analyzer,
            synthesizer=mock_synthesizer, grounder=mock_grounder, evaluator=mock_evaluator,
            store=mock_store, settings=mock_settings,
        )
        report = wf.run("Query")

        assert len(report.workflow_attempts) == 2
        assert report.workflow_attempts[0].triggered_retry is True
        assert report.workflow_attempts[1].triggered_retry is False

    def test_plan_from_prompt_called_on_retry(
        self, mock_planner, mock_researcher, mock_analyzer,
        mock_synthesizer, mock_grounder, mock_evaluator,
        mock_store, mock_settings,
    ):
        """On retry, workflow should call plan_from_prompt, not plan."""
        mock_evaluator.evaluate.side_effect = [
            _make_eval(coverage=0.10, utilization=0.10),  # fail
            _make_eval(coverage=0.90, utilization=0.80),  # pass
        ]
        mock_settings.__dict__["max_workflow_retries"] = 1

        wf = ResearchWorkflow(
            planner=mock_planner, researcher=mock_researcher, analyzer=mock_analyzer,
            synthesizer=mock_synthesizer, grounder=mock_grounder, evaluator=mock_evaluator,
            store=mock_store, settings=mock_settings,
        )
        wf.run("Query")

        # Initial call uses plan(), retry uses plan_from_prompt()
        mock_planner.plan.assert_called_once()
        mock_planner.plan_from_prompt.assert_called_once()

        # The reformulation prompt must contain the failure scores
        reformulation_prompt_used = mock_planner.plan_from_prompt.call_args.args[0]
        assert "0.10" in reformulation_prompt_used  # coverage score

    def test_retry_reason_contains_scores(
        self, mock_planner, mock_researcher, mock_analyzer,
        mock_synthesizer, mock_grounder, mock_evaluator,
        mock_store, mock_settings,
    ):
        mock_evaluator.evaluate.side_effect = [
            _make_eval(coverage=0.30, utilization=0.20),
            _make_eval(coverage=0.90, utilization=0.80),
        ]
        mock_settings.__dict__["max_workflow_retries"] = 1

        wf = ResearchWorkflow(
            planner=mock_planner, researcher=mock_researcher, analyzer=mock_analyzer,
            synthesizer=mock_synthesizer, grounder=mock_grounder, evaluator=mock_evaluator,
            store=mock_store, settings=mock_settings,
        )
        report = wf.run("Query")

        assert report.workflow_attempts[0].retry_reason is not None
        assert "0.30" in report.workflow_attempts[0].retry_reason


# ── Retry exhaustion ──────────────────────────────────────────────────────────

class TestWorkflowRetryExhaustion:
    def test_returns_report_when_retries_exhausted(
        self, mock_planner, mock_researcher, mock_analyzer,
        mock_synthesizer, mock_grounder, mock_evaluator,
        mock_store, mock_settings,
    ):
        """When all retries exhausted, return best result instead of crashing."""
        # Always fail
        mock_evaluator.evaluate.return_value = _make_eval(coverage=0.10, utilization=0.05)
        mock_settings.__dict__["max_workflow_retries"] = 1

        wf = ResearchWorkflow(
            planner=mock_planner, researcher=mock_researcher, analyzer=mock_analyzer,
            synthesizer=mock_synthesizer, grounder=mock_grounder, evaluator=mock_evaluator,
            store=mock_store, settings=mock_settings,
        )
        report = wf.run("Query")

        # Should NOT raise — returns what we have
        assert isinstance(report, AgentReport)
        # Both attempts recorded; second attempt does NOT have triggered_retry=True
        # (because we ran out of retries, not because scores passed)
        assert len(report.workflow_attempts) == 2
