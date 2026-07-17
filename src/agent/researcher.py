"""
src/agent/researcher.py
───────────────────────
ResearcherAgent: retrieves relevant context for a SubQuestion using
HyDE (Hypothetical Document Embeddings) and multi-query search.

For each SubQuestion, it:
1. Searches the HybridStore for each search_query variant in the SubQuestion.
2. Deduplicates results by chunk_id (same chunk may appear in multiple searches).
3. Calls the LLM to synthesize a focused answer from the retrieved contexts.
4. Handles the zero-results case gracefully (no crash, no hallucination).
"""

from __future__ import annotations

import logging
from typing import List

from src.config import Settings
from src.genai.client import GroqClient
from src.genai.prompts import researcher_prompt
from src.models.schemas import ResearchResult, RetrievedContext, SubQuestion
from src.vectorstore.store import HybridStore

logger = logging.getLogger(__name__)

_ZERO_RESULTS_ANSWER = (
    "No relevant information found in the uploaded documents for this sub-question."
)


class ResearcherAgent:
    """
    Retrieves context chunks and synthesizes a focused answer for a SubQuestion.
    """

    def __init__(self, store: HybridStore, client: GroqClient, settings: Settings):
        self._store = store
        self._client = client
        self._default_k = settings.retrieval_k

    def research(self, sub_question: SubQuestion, k: int | None = None) -> ResearchResult:
        """
        Research a single SubQuestion.

        Args:
            sub_question: The SubQuestion (with question + search_queries).
            k: Number of chunks to retrieve per search query. Defaults to
               settings.retrieval_k if not provided.

        Returns:
            ResearchResult with contexts and synthesized answer.
        """
        effective_k = k if k is not None else self._default_k
        logger.info(
            f"ResearcherAgent: researching '{sub_question.question[:60]}…' "
            f"with {len(sub_question.search_queries)} queries, k={effective_k}"
        )

        # Multi-query retrieval — search once per HyDE variant
        seen_chunk_ids: set[str] = set()
        all_contexts: List[RetrievedContext] = []

        for query in sub_question.search_queries:
            results = self._store.search(query, k=effective_k)
            for ctx in results:
                if ctx.chunk.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(ctx.chunk.chunk_id)
                    all_contexts.append(ctx)

        # Zero-results graceful fallback
        if not all_contexts:
            logger.warning(
                f"ResearcherAgent: zero results for '{sub_question.question[:60]}…'. "
                "Returning sentinel answer."
            )
            return ResearchResult(
                sub_question=sub_question,
                contexts=[],
                answer=_ZERO_RESULTS_ANSWER,
                low_confidence=False,
            )

        # Synthesize answer from retrieved contexts
        context_texts = [ctx.chunk.text for ctx in all_contexts]
        prompt = researcher_prompt(sub_question.question, context_texts)
        answer = self._client.generate_text(prompt)

        logger.info(
            f"ResearcherAgent: retrieved {len(all_contexts)} unique contexts, "
            f"answer length={len(answer)} chars"
        )

        return ResearchResult(
            sub_question=sub_question,
            contexts=all_contexts,
            answer=answer,
        )
