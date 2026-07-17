"""
tests/test_planner.py
─────────────────────
Phase 7 test suite: PlannerAgent.

Tests both plan() (builds prompt internally) and plan_from_prompt()
(used by the retry path in workflow.py).
All LLM calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent.planner import PlannerAgent
from src.models.schemas import ResearchPlan, SubQuestion


def _make_mock_plan() -> ResearchPlan:
    return ResearchPlan(
        original_query="What caused the 2008 crisis?",
        reasoning="Decomposed into sub-topics",
        sub_questions=[
            SubQuestion(
                question="What role did mortgage-backed securities play?",
                search_queries=["MBS role 2008 crisis", "mortgage backed securities failure"]
            ),
            SubQuestion(
                question="How did credit rating agencies contribute?",
                search_queries=["credit rating agencies 2008", "rating agency failures"]
            ),
        ]
    )


@pytest.fixture
def mock_gemini_client():
    client = MagicMock()
    client.generate.return_value = _make_mock_plan()
    return client


@pytest.fixture
def planner(mock_gemini_client):
    return PlannerAgent(mock_gemini_client)


class TestPlannerAgent:
    def test_plan_returns_research_plan(self, planner, mock_gemini_client):
        result = planner.plan("What caused the 2008 crisis?")
        assert isinstance(result, ResearchPlan)
        assert result.original_query == "What caused the 2008 crisis?"

    def test_plan_calls_generate_with_correct_schema(self, planner, mock_gemini_client):
        planner.plan("What caused the 2008 crisis?")
        mock_gemini_client.generate.assert_called_once()
        call_args = mock_gemini_client.generate.call_args
        assert call_args.args[1] == ResearchPlan  # second arg is the schema

    def test_plan_prompt_contains_query(self, planner, mock_gemini_client):
        query = "What is the role of central banks?"
        planner.plan(query)
        prompt_used = mock_gemini_client.generate.call_args.args[0]
        assert query in prompt_used

    def test_plan_from_prompt_passes_prompt_directly(self, planner, mock_gemini_client):
        custom_prompt = "Custom reformulation prompt with failure context"
        planner.plan_from_prompt(custom_prompt)
        prompt_used = mock_gemini_client.generate.call_args.args[0]
        assert prompt_used == custom_prompt

    def test_plan_from_prompt_returns_research_plan(self, planner, mock_gemini_client):
        result = planner.plan_from_prompt("Any prompt")
        assert isinstance(result, ResearchPlan)

    def test_plan_and_plan_from_prompt_use_same_schema(self, planner, mock_gemini_client):
        """Both entry points must return the same type — this is the contract
        that lets workflow.py swap between them on retry."""
        planner.plan("Query A")
        schema_from_plan = mock_gemini_client.generate.call_args.args[1]

        mock_gemini_client.reset_mock()

        planner.plan_from_prompt("Custom prompt B")
        schema_from_plan_from_prompt = mock_gemini_client.generate.call_args.args[1]

        assert schema_from_plan == schema_from_plan_from_prompt == ResearchPlan

    def test_plan_sub_questions_non_empty(self, planner):
        result = planner.plan("Some research query")
        assert len(result.sub_questions) > 0

    def test_plan_sub_questions_have_search_queries(self, planner):
        result = planner.plan("Some research query")
        for sq in result.sub_questions:
            assert len(sq.search_queries) >= 1
