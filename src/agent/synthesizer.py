"""
src/agent/synthesizer.py
────────────────────────
SynthesizerAgent: writes the final Markdown research report from
analyzed answers, including ⚠️ disclaimers for low-confidence sections.
"""

from __future__ import annotations

import logging
from typing import List

from src.genai.client import GeminiClient
from src.genai.prompts import synthesizer_prompt
from src.models.schemas import ResearchResult

logger = logging.getLogger(__name__)


class SynthesizerAgent:
    """Synthesizes a complete Markdown research report from analyzed findings."""

    def __init__(self, client: GeminiClient):
        self._client = client

    def synthesize(self, query: str, results: List[ResearchResult]) -> str:
        """
        Write the full Markdown research report.

        Low-confidence sections (where all chunks were below the relevance floor)
        are prefixed with a ⚠️ disclaimer so the reader knows to treat them
        skeptically. The disclaimer is injected before calling the LLM so it
        appears in the prompt context and the LLM preserves it in the output.

        Args:
            query: The original user research question.
            results: Analyzed ResearchResults from the AnalyzerAgent.

        Returns:
            A complete Markdown report string with [^N] footnote citations.

        Raises:
            LLMError: If the LLM call fails after all retries.
        """
        analyzed_answers: List[str] = []

        for result in results:
            if result.low_confidence:
                prefix = (
                    "⚠️ LOW CONFIDENCE — all retrieved sources for this sub-question "
                    "scored below the relevance threshold. This section may be unreliable.\n"
                )
                analyzed_answers.append(prefix + result.answer)
            else:
                analyzed_answers.append(result.answer)

        prompt = synthesizer_prompt(query, analyzed_answers)
        logger.info(
            f"SynthesizerAgent: synthesizing report for query '{query[:60]}' "
            f"from {len(results)} findings"
        )
        report = self._client.generate_text(prompt, temperature=0.4)
        logger.info(f"SynthesizerAgent: report generated ({len(report)} chars)")
        return report
