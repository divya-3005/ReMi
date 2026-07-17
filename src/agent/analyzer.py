"""
src/agent/analyzer.py
─────────────────────
AnalyzerAgent: scores retrieved chunk relevance using Groq (fast inference)
and filters chunks below the relevance floor.

Uses the GroqClient (not Gemini) because:
- Relevance scoring is a simple classification task; frontier model quality
  is unnecessary and Groq's throughput is significantly higher.
- The Analyzer runs on every sub-question result — using Groq keeps the
  per-query latency reasonable.

Edge case — all chunks below floor:
Instead of returning an empty list (which would cause the Synthesizer to
produce an empty section and confuse the Evaluator), we keep the single
highest-scoring chunk and set low_confidence=True on the ResearchResult.
The Synthesizer prompt checks this flag and adds a ⚠️ disclaimer.
"""

from __future__ import annotations

import logging
from typing import List

from pydantic import BaseModel

from src.config import Settings
from src.genai.client import GroqClient
from src.genai.prompts import analyzer_prompt
from src.models.schemas import ResearchResult, RetrievedContext

logger = logging.getLogger(__name__)


class _ScoreResponse(BaseModel):
    """Internal schema for the Groq analyzer response."""
    scores: List[float]


class AnalyzerAgent:
    """
    Scores and filters retrieved chunks by relevance to their sub-question.
    """

    def __init__(self, client: GroqClient, settings: Settings):
        self._client = client
        # Settings stored for future use (e.g., per-agent logging config).
        # The relevance_floor is passed per-call so workflow.py controls it.
        self._settings = settings

    def analyze(
        self,
        results: List[ResearchResult],
        relevance_floor: float,
    ) -> List[ResearchResult]:
        """
        Score and filter the contexts in each ResearchResult.

        Args:
            results: List of ResearchResults from the ResearcherAgent.
            relevance_floor: Minimum relevance score to keep a chunk.
                             Must be passed explicitly — not read from Settings —
                             because the workflow always passes settings.analyzer_relevance_floor.
                             This makes the call site self-documenting and prevents
                             accidental omission.

        Returns:
            A new list of ResearchResults with contexts filtered by relevance.
            Each result in the output corresponds to the same-indexed input result.
        """
        filtered_results: List[ResearchResult] = []

        for result in results:
            if not result.contexts:
                # Nothing to score — pass through unchanged
                filtered_results.append(result)
                continue

            # Score all chunks for this sub-question
            candidate_texts = [ctx.chunk.text for ctx in result.contexts]
            prompt = analyzer_prompt(result.sub_question.question, candidate_texts)
            score_response = self._client.generate(prompt, _ScoreResponse)
            scores = score_response.scores

            # Guard against score count mismatch (LLM occasionally drops a score)
            if len(scores) != len(result.contexts):
                logger.warning(
                    f"AnalyzerAgent: score count mismatch for '{result.sub_question.question[:40]}'. "
                    f"Expected {len(result.contexts)}, got {len(scores)}. "
                    "Padding with 0.0."
                )
                scores = scores[:len(result.contexts)]
                scores.extend([0.0] * (len(result.contexts) - len(scores)))

            # Filter by relevance floor
            scored_contexts = list(zip(scores, result.contexts))
            passing = [
                (score, ctx)
                for score, ctx in scored_contexts
                if score >= relevance_floor
            ]

            low_confidence = False

            if passing:
                filtered_contexts = [ctx for _, ctx in passing]
            else:
                # All below floor — keep the single best to prevent empty sections
                best_score, best_ctx = max(scored_contexts, key=lambda t: t[0])
                filtered_contexts = [best_ctx]
                low_confidence = True
                logger.warning(
                    f"AnalyzerAgent: all {len(result.contexts)} chunks below floor "
                    f"({relevance_floor}) for '{result.sub_question.question[:40]}'. "
                    f"Keeping best chunk (score={best_score:.2f}), low_confidence=True."
                )

            logger.info(
                f"AnalyzerAgent: '{result.sub_question.question[:40]}' "
                f"→ {len(result.contexts)} contexts → {len(filtered_contexts)} kept "
                f"(floor={relevance_floor}, low_confidence={low_confidence})"
            )

            filtered_results.append(
                ResearchResult(
                    sub_question=result.sub_question,
                    contexts=filtered_contexts,
                    answer=result.answer,
                    low_confidence=low_confidence,
                )
            )

        return filtered_results
