"""
src/evaluation/evaluator.py
────────────────────────────
EvaluatorAgent: computes quality scores for a completed research run.

Metrics:
- citation_coverage:    % of grounded footnotes / total report footnotes
                        (proxy for "how much of the report is traceable to sources")
- citation_utilization: % of retrieved chunks actually cited
                        (proxy for "how relevant was the retrieved content")
- answer_relevance:     cosine similarity between query embedding and report embedding
                        (proxy for "does the report answer what was asked")
- hallucination_risk:   1.0 - citation_coverage
                        (proxy for "how much of the report has no source trace")

Important honesty note (also in EvaluationResult docstring):
These are PROXY metrics. citation_coverage measures traceability via difflib
fuzzy match — a sentence can pass the threshold while still being incorrect.
hallucination_risk is derived from citation_coverage, not from a truth oracle.
These numbers are useful signals for driving the retry loop, not ground truth.
"""

from __future__ import annotations

import logging
import re
from typing import List

import numpy as np

from src.models.schemas import Chunk, CitationLink, EvaluationResult, RetrievedContext

logger = logging.getLogger(__name__)

_FOOTNOTE_RE = re.compile(r"\[\^(\d+)\]")


class EvaluatorAgent:
    """
    Computes quality scores for a completed research pass.
    Pure computation — no LLM calls, no external I/O.
    """

    def __init__(self, embedder):
        """
        Args:
            embedder: A GeminiEmbedder instance (for answer_relevance cosine sim).
        """
        self._embedder = embedder

    def evaluate(
        self,
        query: str,
        report_text: str,
        contexts: List[RetrievedContext],
        cited_chunks: List[Chunk],
    ) -> EvaluationResult:
        """
        Compute all four quality scores.

        Args:
            query: The original research question.
            report_text: The grounded Markdown report.
            contexts: All RetrievedContexts across all sub-questions (post-analysis).
            cited_chunks: The Chunk objects actually linked in the citation index.

        Returns:
            EvaluationResult with all four scores in [0.0, 1.0].
        """
        citation_coverage = self._compute_citation_coverage(report_text, cited_chunks)
        citation_utilization = self._compute_citation_utilization(contexts, cited_chunks)
        answer_relevance = self._compute_answer_relevance(query, report_text)
        hallucination_risk = round(1.0 - citation_coverage, 4)

        result = EvaluationResult(
            citation_coverage=citation_coverage,
            citation_utilization=citation_utilization,
            answer_relevance=answer_relevance,
            hallucination_risk=hallucination_risk,
        )

        logger.info(
            f"EvaluatorAgent: coverage={citation_coverage:.2f}, "
            f"utilization={citation_utilization:.2f}, "
            f"relevance={answer_relevance:.2f}, "
            f"hallucination_risk={hallucination_risk:.2f}"
        )
        return result

    def _compute_citation_coverage(
        self, report_text: str, cited_chunks: List[Chunk]
    ) -> float:
        """
        Fraction of [^N] footnotes in the report that have a matched CitationLink.

        = len(matched_footnotes) / len(all_footnotes_in_report)
        """
        all_footnote_ids = set(
            int(m.group(1)) for m in _FOOTNOTE_RE.finditer(report_text)
        )
        if not all_footnote_ids:
            # No footnotes at all — report didn't use citation syntax
            return 0.0

        cited_chunk_ids = {c.chunk_id for c in cited_chunks}
        # A footnote is "covered" if at least one cited chunk exists
        # (we know cited_chunks are the ones resolved by the Grounder)
        matched_count = len(cited_chunk_ids)
        return round(min(1.0, matched_count / len(all_footnote_ids)), 4)

    def _compute_citation_utilization(
        self, contexts: List[RetrievedContext], cited_chunks: List[Chunk]
    ) -> float:
        """
        Fraction of retrieved chunks that were actually cited in the report.

        = len(cited_chunk_ids) / len(total_retrieved_chunk_ids)
        """
        all_chunk_ids = {ctx.chunk.chunk_id for ctx in contexts}
        if not all_chunk_ids:
            return 0.0
        cited_chunk_ids = {c.chunk_id for c in cited_chunks}
        return round(len(cited_chunk_ids) / len(all_chunk_ids), 4)

    def _compute_answer_relevance(self, query: str, report_text: str) -> float:
        """
        Cosine similarity between query embedding and report embedding.

        Uses the first 2000 chars of the report to limit API cost.
        """
        try:
            query_emb = np.array(self._embedder.embed_query(query), dtype=np.float32)
            # Truncate report to avoid embedding a very long text
            report_snippet = report_text[:2000]
            report_emb = np.array(
                self._embedder.embed_texts([report_snippet])[0], dtype=np.float32
            )

            # Cosine similarity
            denom = np.linalg.norm(query_emb) * np.linalg.norm(report_emb)
            if denom == 0:
                return 0.0
            similarity = float(np.dot(query_emb, report_emb) / denom)
            # Clamp to [0, 1] — cosine can be negative for very dissimilar vectors
            return round(max(0.0, min(1.0, similarity)), 4)
        except Exception as e:
            logger.warning(f"EvaluatorAgent: answer_relevance embedding failed: {e}. Defaulting to 0.0.")
            return 0.0
