"""
tests/test_prompts.py
─────────────────────
Phase 6 test suite: prompt-building functions.

Prompts are pure string functions — no LLM calls, no mocking needed.
Tests verify structural guarantees: required content present, formatting
correct, failure context included in reformulation prompts.
"""

from __future__ import annotations

from src.genai.prompts import (
    analyzer_prompt,
    hyde_prompt,
    planner_prompt,
    reformulation_prompt,
    researcher_prompt,
    synthesizer_prompt,
)
from src.models.schemas import EvaluationResult


# ── planner_prompt ────────────────────────────────────────────────────────────

class TestPlannerPrompt:
    def test_contains_query(self):
        prompt = planner_prompt("What caused the 2008 financial crisis?")
        assert "What caused the 2008 financial crisis?" in prompt

    def test_contains_json_instruction(self):
        prompt = planner_prompt("Any query")
        # Must instruct the LLM to return JSON with the expected field names
        assert "sub_questions" in prompt
        assert "search_queries" in prompt

    def test_contains_num_sub_questions(self):
        prompt = planner_prompt("Query", num_sub_questions=4)
        assert "4" in prompt

    def test_default_num_sub_questions(self):
        prompt = planner_prompt("Query")
        assert "3" in prompt


# ── hyde_prompt ───────────────────────────────────────────────────────────────

class TestHydePrompt:
    def test_contains_sub_question(self):
        prompt = hyde_prompt("What is the role of CDOs in the 2008 crisis?")
        assert "CDOs" in prompt

    def test_instructs_hypothetical_passage(self):
        prompt = hyde_prompt("Any question")
        lower = prompt.lower()
        assert "hypothetical" in lower or "passage" in lower or "document" in lower


# ── researcher_prompt ─────────────────────────────────────────────────────────

class TestResearcherPrompt:
    def test_contains_sub_question(self):
        prompt = researcher_prompt(
            sub_question="What is quantitative easing?",
            context_texts=["Central banks buy assets.", "Money supply increases."]
        )
        assert "quantitative easing" in prompt.lower()

    def test_contains_all_context_texts(self):
        contexts = ["Context A.", "Context B.", "Context C."]
        prompt = researcher_prompt("Q?", contexts)
        for ctx in contexts:
            assert ctx in prompt

    def test_empty_context_handled_gracefully(self):
        prompt = researcher_prompt("Q?", [])
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ── analyzer_prompt ───────────────────────────────────────────────────────────

class TestAnalyzerPrompt:
    def test_contains_sub_question(self):
        prompt = analyzer_prompt(
            sub_question="How did Lehman Brothers collapse?",
            candidates=["Lehman filed for bankruptcy."]
        )
        assert "Lehman Brothers" in prompt

    def test_instructs_relevance_scoring(self):
        prompt = analyzer_prompt("Q?", ["Some chunk text."])
        lower = prompt.lower()
        assert "relevance" in lower or "score" in lower or "relevant" in lower

    def test_contains_candidates(self):
        candidates = ["Chunk alpha.", "Chunk beta."]
        prompt = analyzer_prompt("Q?", candidates)
        for c in candidates:
            assert c in prompt


# ── synthesizer_prompt ────────────────────────────────────────────────────────

class TestSynthesizerPrompt:
    def test_contains_query(self):
        prompt = synthesizer_prompt(
            query="Explain the role of CDOs.",
            analyzed_answers=["CDOs are complex instruments.", "They were widely mis-rated."]
        )
        assert "CDOs" in prompt

    def test_contains_analyzed_answers(self):
        answers = ["Answer A.", "Answer B."]
        prompt = synthesizer_prompt("Query", answers)
        for a in answers:
            assert a in prompt

    def test_instructs_citation_format(self):
        prompt = synthesizer_prompt("Q?", ["A."])
        # Must tell the LLM to use inline citations so the grounder can link them
        lower = prompt.lower()
        assert "citation" in lower or "[^" in prompt or "footnote" in lower


# ── reformulation_prompt ──────────────────────────────────────────────────────

class TestReformulationPrompt:
    def _make_failed_eval(self) -> EvaluationResult:
        return EvaluationResult(
            citation_coverage=0.30,
            citation_utilization=0.20,
            answer_relevance=0.70,
            hallucination_risk=0.70,
        )

    def test_contains_original_query(self):
        prompt = reformulation_prompt(
            query="What caused the housing bubble?",
            previous_sub_questions=["Q1", "Q2"],
            failed_eval=self._make_failed_eval(),
        )
        assert "housing bubble" in prompt.lower()

    def test_contains_previous_sub_questions(self):
        prompt = reformulation_prompt(
            query="Q",
            previous_sub_questions=["Previous question one.", "Previous question two."],
            failed_eval=self._make_failed_eval(),
        )
        assert "Previous question one." in prompt
        assert "Previous question two." in prompt

    def test_contains_failed_score_values(self):
        """The reformulation prompt MUST include the actual scores so the LLM
        understands why the previous attempt failed."""
        ev = self._make_failed_eval()
        prompt = reformulation_prompt(
            query="Q",
            previous_sub_questions=["Q1"],
            failed_eval=ev,
        )
        # Both scores that trigger retry should appear in the prompt
        assert "0.30" in prompt or "30" in prompt  # citation_coverage
        assert "0.20" in prompt or "20" in prompt  # citation_utilization

    def test_instructs_new_sub_questions(self):
        prompt = reformulation_prompt(
            query="Q",
            previous_sub_questions=["Q1"],
            failed_eval=self._make_failed_eval(),
        )
        lower = prompt.lower()
        # Must ask for new, different sub-questions
        assert "sub_question" in lower or "reformulat" in lower or "new" in lower
