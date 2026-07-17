"""
src/agent/workflow.py
─────────────────────
ResearchWorkflow: orchestrates the full multi-agent research pipeline.

The agentic feedback loop:

  ┌──────────────────────────────────────────────────────┐
  │  attempt = 0                                         │
  │  last_evaluation = None                              │
  │                                                      │
  │  while attempt <= max_workflow_retries:              │
  │    if attempt == 0:                                  │
  │      plan = Planner.plan(query)                      │
  │    else:                                             │
  │      prompt = reformulation_prompt(                  │
  │          query, prev_sub_questions, last_evaluation) │
  │      plan = Planner.plan_from_prompt(prompt)         │
  │                                                      │
  │    research_results = [Researcher.research(sq)       │
  │                        for sq in plan.sub_questions] │
  │    analyzed = Analyzer.analyze(research_results,     │
  │                                relevance_floor)      │
  │    report_text = Synthesizer.synthesize(query,       │
  │                                        analyzed)     │
  │    report_text, citations = Grounder.ground(         │
  │                                report_text, chunks)  │
  │    contexts = flatten(r.contexts for r in analyzed)  │
  │    cited_chunks = [c.chunk for c in contexts         │
  │                    if c.chunk.chunk_id in            │
  │                       {cit.chunk_id for cit in       │
  │                        citations}]                   │
  │    evaluation = Evaluator.evaluate(                  │
  │         query, report_text, contexts, cited_chunks)  │
  │                                                      │
  │    triggered_retry = (                               │
  │      evaluation.citation_coverage < min_coverage AND │
  │      evaluation.citation_utilization < min_util AND  │
  │      attempt < max_workflow_retries                  │
  │    )                                                 │
  │    record WorkflowAttempt                            │
  │    if not triggered_retry: break                     │
  │    attempt += 1                                      │
  │    last_evaluation = evaluation                      │
  │  end while                                           │
  │                                                      │
  │  return AgentReport(...)                             │
  └──────────────────────────────────────────────────────┘

Key design decisions:
- `start_time` is set before the loop; `last_evaluation` initialized to None.
- The relevance_floor passed to Analyzer.analyze() is ALWAYS the fixed value
  from Settings — it never changes between retries. Query reformulation is the
  retry lever, not filter relaxation.
- On retry, the previous sub-question strings are extracted from the last plan
  and passed to reformulation_prompt() verbatim, so the LLM can explicitly
  avoid repeating them.
- If all retries are exhausted and scores still fail, we return the final
  attempt's result rather than raising — the caller gets a real AgentReport
  with low scores, not a crash.
"""

from __future__ import annotations

import logging
import time
from typing import List

from src.agent.analyzer import AnalyzerAgent
from src.agent.planner import PlannerAgent
from src.agent.researcher import ResearcherAgent
from src.agent.synthesizer import SynthesizerAgent
from src.config import Settings
from src.evaluation.evaluator import EvaluatorAgent
from src.genai.prompts import reformulation_prompt
from src.grounding.grounder import GrounderAgent
from src.models.schemas import (
    AgentReport,
    Chunk,
    EvaluationResult,
    ResearchPlan,
    RetrievedContext,
    WorkflowAttempt,
)
from src.vectorstore.store import HybridStore

logger = logging.getLogger(__name__)


class ResearchWorkflow:
    """Orchestrates the full multi-agent research pipeline with agentic retry."""

    # Limitations applied in every run (surfaced in AgentReport for transparency)
    KNOWN_LIMITATIONS = [
        "difflib_grounding",           # citation matching is fuzzy, not truth-verified
        "cosine_relevance_proxy",      # answer_relevance is cosine sim, not a truth metric
    ]

    def __init__(
        self,
        planner: PlannerAgent,
        researcher: ResearcherAgent,
        analyzer: AnalyzerAgent,
        synthesizer: SynthesizerAgent,
        grounder: GrounderAgent,
        evaluator: EvaluatorAgent,
        store: HybridStore,
        settings: Settings,
    ):
        self._planner = planner
        self._researcher = researcher
        self._analyzer = analyzer
        self._synthesizer = synthesizer
        self._grounder = grounder
        self._evaluator = evaluator
        self._store = store
        self._settings = settings

    def run(self, query: str) -> AgentReport:
        """
        Execute the full research pipeline with agentic retry.

        Args:
            query: The user's research question.

        Returns:
            AgentReport with the answer, citations, evaluation scores,
            and a full WorkflowAttempt audit trail.
        """
        start_time = time.time()
        last_evaluation: EvaluationResult | None = None
        last_plan: ResearchPlan | None = None
        workflow_attempts: List[WorkflowAttempt] = []

        # Final values — set on every attempt, so they always hold the last pass's data
        final_report_text = ""
        final_citations = []
        final_evaluation: EvaluationResult | None = None

        for attempt in range(self._settings.max_workflow_retries + 1):
            logger.info(
                f"ResearchWorkflow: attempt {attempt + 1}/"
                f"{self._settings.max_workflow_retries + 1} for query '{query[:60]}'"
            )

            # ── Step 1: Plan ──────────────────────────────────────────────────
            if attempt == 0:
                plan = self._planner.plan(query)
            else:
                # Build the reformulation prompt with failure context from last attempt
                prev_sub_questions = [sq.question for sq in last_plan.sub_questions]
                reformat_prompt = reformulation_prompt(
                    query=query,
                    previous_sub_questions=prev_sub_questions,
                    failed_eval=last_evaluation,
                )
                plan = self._planner.plan_from_prompt(reformat_prompt)

            last_plan = plan

            # ── Step 2: Research (parallel-ready, sequential for now) ─────────
            research_results = [
                self._researcher.research(sq)
                for sq in plan.sub_questions
            ]

            # ── Step 3: Analyze + filter ──────────────────────────────────────
            analyzed_results = self._analyzer.analyze(
                research_results,
                relevance_floor=self._settings.analyzer_relevance_floor,
            )

            # ── Step 4: Synthesize ────────────────────────────────────────────
            report_text = self._synthesizer.synthesize(query, analyzed_results)

            # ── Step 5: Ground citations ──────────────────────────────────────
            # Gather all chunks available in the store for grounding
            all_store_chunks: List[Chunk] = list(self._store.chunks)
            report_text, citations = self._grounder.ground(report_text, all_store_chunks)

            # ── Step 6: Derive contexts and cited_chunks ──────────────────────
            # contexts = all RetrievedContexts across all analyzed sub-questions
            contexts: List[RetrievedContext] = [
                ctx
                for result in analyzed_results
                for ctx in result.contexts
            ]

            # cited_chunks = only the chunks that the Grounder successfully linked
            cited_chunk_ids = {cit.chunk_id for cit in citations}
            cited_chunks: List[Chunk] = [
                ctx.chunk
                for ctx in contexts
                if ctx.chunk.chunk_id in cited_chunk_ids
            ]

            # ── Step 7: Evaluate ──────────────────────────────────────────────
            evaluation = self._evaluator.evaluate(
                query=query,
                report_text=report_text,
                contexts=contexts,
                cited_chunks=cited_chunks,
            )

            # ── Step 8: Quality gate — decide whether to retry ────────────────
            scores_pass = (
                evaluation.citation_coverage >= self._settings.min_citation_coverage
                and evaluation.citation_utilization >= self._settings.min_citation_utilization
            )
            more_retries_available = attempt < self._settings.max_workflow_retries
            triggered_retry = (not scores_pass) and more_retries_available

            retry_reason: str | None = None
            if triggered_retry:
                retry_reason = (
                    f"Quality gates failed: "
                    f"citation_coverage={evaluation.citation_coverage:.2f} "
                    f"(min={self._settings.min_citation_coverage}), "
                    f"citation_utilization={evaluation.citation_utilization:.2f} "
                    f"(min={self._settings.min_citation_utilization}). "
                    "Reformulating sub-questions."
                )
                logger.warning(f"ResearchWorkflow attempt {attempt + 1}: {retry_reason}")

            workflow_attempts.append(
                WorkflowAttempt(
                    attempt_number=attempt,
                    evaluation=evaluation,
                    triggered_retry=triggered_retry,
                    retry_reason=retry_reason,
                )
            )

            # Store final state of this attempt
            final_report_text = report_text
            final_citations = citations
            final_evaluation = evaluation
            last_evaluation = evaluation

            if not triggered_retry:
                if scores_pass:
                    logger.info(
                        f"ResearchWorkflow: quality gates passed on attempt {attempt + 1}."
                    )
                else:
                    logger.warning(
                        f"ResearchWorkflow: quality gates still failing after all retries. "
                        "Returning best available result."
                    )
                break

        elapsed = time.time() - start_time

        return AgentReport(
            query=query,
            answer_text=final_report_text,
            citations=final_citations,
            evaluation=final_evaluation,
            workflow_attempts=workflow_attempts,
            elapsed_seconds=elapsed,
            known_limitations_applied=list(self.KNOWN_LIMITATIONS),
        )
