"""
src/models/schemas.py
─────────────────────
All Pydantic data models for ReMi.

These are the source of truth for data shapes across every layer of the system.
Changing a field here is the only place a change needs to happen — all other
modules import from here.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


# ── Document layer ────────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """
    Metadata for an ingested document.

    `doc_id` is auto-generated as a UUID4 string if not provided.
    `uploaded_at` is set to the current UTC time on construction.
    """

    doc_id: str = Field(default_factory=_new_uuid)
    filename: str
    file_type: Literal["pdf", "txt"]
    num_pages: int
    uploaded_at: datetime = Field(default_factory=_utcnow)


class Chunk(BaseModel):
    """
    A single text chunk produced by the chunker.

    `char_start` and `char_end` are byte-exact offsets into the original
    document text. They are mandatory — the grounding layer requires them
    to map report claims back to source character spans.
    """

    chunk_id: str = Field(default_factory=_new_uuid)
    doc_id: str
    text: str
    page_number: int
    char_start: int  # NOT optional — grounding depends on this
    char_end: int    # NOT optional — grounding depends on this
    embedding: Optional[list[float]] = None


# ── Agent input/output layer ──────────────────────────────────────────────────

class SubQuestion(BaseModel):
    """
    A decomposed sub-question produced by the Planner agent.

    `search_queries` contains 1–3 HyDE (Hypothetical Document Embedding)
    variants of the sub-question, used to broaden retrieval coverage.
    At least one search query is required.
    """

    question: str
    search_queries: list[str] = Field(min_length=1)

    @field_validator("search_queries")
    @classmethod
    def at_least_one_query(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("search_queries must contain at least one entry")
        return v


class ResearchPlan(BaseModel):
    """
    The structured research plan produced by the Planner agent.

    `sub_questions` must be non-empty — a plan with no questions is useless
    and likely indicates an LLM output failure that should be retried.
    """

    original_query: str
    reasoning: str
    sub_questions: list[SubQuestion] = Field(min_length=1)

    @field_validator("sub_questions")
    @classmethod
    def at_least_one_sub_question(cls, v: list[SubQuestion]) -> list[SubQuestion]:
        if not v:
            raise ValueError("sub_questions must contain at least one entry")
        return v


class RetrievedContext(BaseModel):
    """A single retrieved chunk, with its retrieval score and method."""

    chunk: Chunk
    score: float
    retrieval_method: Literal["dense", "sparse", "rrf"]


class ResearchResult(BaseModel):
    """
    The output of one ResearcherAgent run for a single SubQuestion.

    `low_confidence` is set True by the AnalyzerAgent when all retrieved
    chunks scored below the relevance floor. The SynthesizerAgent will
    include a ⚠️ disclaimer for low-confidence sections.
    """

    sub_question: SubQuestion
    contexts: list[RetrievedContext]
    answer: str
    low_confidence: bool = False


# ── Evaluation layer ──────────────────────────────────────────────────────────

class EvaluationResult(BaseModel):
    """
    Quality scores for a completed research run.

    IMPORTANT — what these metrics actually measure:
    - citation_coverage: % of report sentences that found a matching source span
      via difflib fuzzy match (threshold 0.6). This is a COVERAGE metric, not a
      truth metric. A sentence can be cited while still being wrong.
    - citation_utilization: % of retrieved chunks actually referenced in the report.
      Named to avoid confusion with the RAGAS "context_precision" metric, which
      means something different.
    - answer_relevance: cosine similarity between query embedding and report embedding.
    - hallucination_risk: 1.0 - citation_coverage. Proxy only; read the docstring
      before quoting this number in an interview.

    All scores are in [0.0, 1.0].
    """

    citation_coverage: float
    citation_utilization: float
    answer_relevance: float
    hallucination_risk: float

    @field_validator(
        "citation_coverage",
        "citation_utilization",
        "answer_relevance",
        "hallucination_risk",
    )
    @classmethod
    def score_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Score must be in [0.0, 1.0], got {v}")
        return v


# ── Citation layer ────────────────────────────────────────────────────────────

class CitationLink(BaseModel):
    """
    Links a footnote in the report back to a character span in a source chunk.

    `excerpt` is truncated to 120 characters for display in the UI's SourceDrawer.
    """

    footnote_id: int
    chunk_id: str
    char_start: int
    char_end: int
    excerpt: str

    @field_validator("excerpt")
    @classmethod
    def truncate_excerpt(cls, v: str) -> str:
        """Enforce max 120-char display length."""
        return v[:120]


# ── Workflow audit layer ──────────────────────────────────────────────────────

class WorkflowAttempt(BaseModel):
    """
    Records the outcome of one pass through the full agent pipeline.

    Stored in AgentReport.workflow_attempts so the UI's WorkflowTrace
    component can show whether a retry happened and why.
    """

    attempt_number: int
    evaluation: EvaluationResult
    triggered_retry: bool
    retry_reason: Optional[str] = None


class AgentReport(BaseModel):
    """
    The final output of a complete ResearchWorkflow run.

    `known_limitations_applied` is a list of string keys documenting which
    known limitations affected this specific run (e.g. "difflib_grounding",
    "retry_assumes_filter_failure_not_corpus_gap"). Surfaced in the UI and
    logged for transparency.
    """

    query: str
    answer_text: str
    citations: list[CitationLink]
    evaluation: EvaluationResult
    workflow_attempts: list[WorkflowAttempt]
    elapsed_seconds: float
    known_limitations_applied: list[str]

    @field_validator("elapsed_seconds")
    @classmethod
    def elapsed_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"elapsed_seconds must be non-negative, got {v}")
        return v
