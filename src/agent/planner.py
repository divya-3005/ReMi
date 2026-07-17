"""
src/agent/planner.py
────────────────────
PlannerAgent: decomposes a user query into a structured ResearchPlan.

Exposes two entry points:
  plan(query)                 — initial call; builds the prompt internally
  plan_from_prompt(prompt)    — retry path; the workflow builds the full
                                reformulation_prompt() and passes it here,
                                bypassing internal prompt construction.

This separation is the contract that makes the agentic retry loop work:
the workflow controls exactly what the Planner sees on retry (failure scores,
previous sub-questions, new instructions) without the Planner needing to
know it's retrying.
"""

from __future__ import annotations

import logging

from src.genai.client import GeminiClient
from src.genai.prompts import planner_prompt
from src.models.schemas import ResearchPlan

logger = logging.getLogger(__name__)


class PlannerAgent:
    """Decomposes a research query into a structured ResearchPlan."""

    def __init__(self, client: GeminiClient):
        self._client = client

    def plan(self, query: str, num_sub_questions: int = 3) -> ResearchPlan:
        """
        Decompose the query into a ResearchPlan.

        Builds a planner_prompt() internally and calls the LLM.

        Args:
            query: The original user research question.
            num_sub_questions: How many sub-questions to generate (default 3).

        Returns:
            A validated ResearchPlan Pydantic model.

        Raises:
            LLMError: If the LLM call fails after all retries.
        """
        prompt = planner_prompt(query, num_sub_questions=num_sub_questions)
        logger.info(f"PlannerAgent.plan: decomposing query into {num_sub_questions} sub-questions")
        plan = self._client.generate(prompt, ResearchPlan)
        logger.info(
            f"PlannerAgent.plan: generated {len(plan.sub_questions)} sub-questions"
        )
        return plan

    def plan_from_prompt(self, prompt: str) -> ResearchPlan:
        """
        Generate a ResearchPlan from a pre-built prompt.

        Used by the retry path in workflow.py, where the workflow has already
        constructed a reformulation_prompt() that includes failure context.
        The Planner skips internal prompt building and passes the provided
        prompt directly to the LLM.

        Args:
            prompt: A fully constructed prompt string (e.g. from reformulation_prompt()).

        Returns:
            A validated ResearchPlan Pydantic model.

        Raises:
            LLMError: If the LLM call fails after all retries.
        """
        logger.info("PlannerAgent.plan_from_prompt: planning from pre-built reformulation prompt")
        plan = self._client.generate(prompt, ResearchPlan)
        logger.info(
            f"PlannerAgent.plan_from_prompt: generated {len(plan.sub_questions)} new sub-questions"
        )
        return plan
