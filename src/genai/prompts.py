"""
src/genai/prompts.py
────────────────────
Pure prompt-building functions for every agent in the pipeline.

Design decisions:
- Every function is a pure function (str → str). No LLM calls here.
- Prompts are explicit about the JSON schema they expect, using the exact
  Pydantic field names from schemas.py. This makes parsing deterministic.
- The reformulation_prompt includes exact score values from the failed
  EvaluationResult so the LLM understands the specific failure mode.
- All prompts instruct the LLM to cite using [^N] footnote syntax so the
  GrounderAgent can parse them via difflib matching.
"""

from __future__ import annotations

from typing import List

from src.models.schemas import EvaluationResult


def planner_prompt(query: str, num_sub_questions: int = 3) -> str:
    """
    Instruct the PlannerAgent to decompose a query into sub-questions.

    Returns a prompt that will cause the LLM to emit a JSON object matching
    the ResearchPlan schema.
    """
    return f"""You are an expert research planner. Your job is to decompose a complex research
query into {num_sub_questions} focused sub-questions that together will fully answer the
original query when answered individually.

For each sub-question, generate 2-3 alternative search queries (HyDE variants) that could
retrieve hypothetical document passages answering that sub-question. These variants help
cast a wider retrieval net.

Original query: {query}

Respond with ONLY a JSON object matching this exact schema (no markdown, no explanation):
{{
  "original_query": "<the original query>",
  "reasoning": "<1-2 sentences explaining your decomposition strategy>",
  "sub_questions": [
    {{
      "question": "<focused sub-question>",
      "search_queries": ["<query variant 1>", "<query variant 2>", "<query variant 3>"]
    }}
  ]
}}

Generate exactly {num_sub_questions} sub_questions."""


def hyde_prompt(sub_question: str) -> str:
    """
    Generate a hypothetical document passage for a sub-question (HyDE).

    HyDE (Hypothetical Document Embeddings) improves dense retrieval by generating
    a plausible answer, then using its embedding instead of the raw question
    embedding. The generated passage is never shown to the user.
    """
    return f"""Write a short hypothetical document passage (2-3 sentences) that would
perfectly answer the following question if it existed in a real document.

Question: {sub_question}

Write ONLY the hypothetical passage. No preamble, no explanation, no attribution.
The passage should read as if extracted from a real document on this topic."""


def researcher_prompt(sub_question: str, context_texts: List[str]) -> str:
    """
    Instruct the ResearcherAgent to synthesize an answer from retrieved context.
    """
    if not context_texts:
        contexts_block = "[No relevant context was retrieved for this sub-question.]"
    else:
        contexts_block = "\n\n".join(
            f"[Source {i+1}]\n{text}" for i, text in enumerate(context_texts)
        )

    return f"""You are a precise research analyst. Using ONLY the provided source passages,
answer the following sub-question. Do not add information not present in the sources.
If the sources do not contain enough information to answer the question, say so explicitly.

Sub-question: {sub_question}

Source passages:
{contexts_block}

Provide a concise, factual answer (2-4 sentences) based strictly on the source passages above."""


def analyzer_prompt(sub_question: str, candidates: List[str]) -> str:
    """
    Instruct the AnalyzerAgent to score the relevance of retrieved chunks.

    Returns a prompt that will cause the LLM to emit a JSON object with
    relevance scores for each candidate chunk.
    """
    candidates_block = "\n\n".join(
        f"[Chunk {i}]\n{text}" for i, text in enumerate(candidates)
    )

    return f"""You are a relevance scoring expert. For each text chunk below, score its
relevance to the sub-question on a scale of 0.0 to 1.0.

- 1.0 = directly answers the sub-question
- 0.7 = strongly related, provides useful supporting context
- 0.4 = tangentially related, some useful information
- 0.1 = mostly unrelated, minimal relevance
- 0.0 = completely irrelevant

Sub-question: {sub_question}

Text chunks to score:
{candidates_block}

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "scores": [<score for chunk 0>, <score for chunk 1>, ...]
}}

The "scores" array must have exactly {len(candidates)} elements."""


def synthesizer_prompt(query: str, analyzed_answers: List[str]) -> str:
    """
    Instruct the SynthesizerAgent to write the final research report.

    The report uses [^N] footnote citation syntax. Each sentence making a
    factual claim should end with one or more [^N] citations referencing
    the source answer that supports it. The GrounderAgent will resolve these
    footnotes to exact character spans in source chunks.
    """
    answers_block = "\n\n".join(
        f"[Research Finding {i+1}]\n{answer}" for i, answer in enumerate(analyzed_answers)
    )

    return f"""You are a senior research analyst writing a comprehensive report.
Using the research findings below, write a well-structured, detailed Markdown report
that fully answers the original query.

CITATION RULES (strictly follow):
1. Every factual claim MUST be followed by a footnote citation: [^1], [^2], etc.
2. Number citations sequentially starting from [^1].
3. The same source can be cited multiple times with the same number.
4. Do not fabricate information — only use what is in the research findings.
5. Sections with low-confidence findings (marked ⚠️) must include a disclaimer.

Original query: {query}

Research findings:
{answers_block}

Write the complete Markdown report now. Start with a clear title (# heading).
End with a ## References section listing each [^N] footnote."""


def reformulation_prompt(
    query: str,
    previous_sub_questions: List[str],
    failed_eval: EvaluationResult,
) -> str:
    """
    Instruct the PlannerAgent to reformulate sub-questions after a failed attempt.

    This is the core of the agentic feedback loop. The failed EvaluationResult
    scores are included verbatim so the LLM understands exactly WHY the previous
    attempt failed and can generate substantially different sub-questions.

    Failure mode context:
    - Low citation_coverage → report claims weren't backed by retrieved sources.
      Cause: sub-questions didn't match what the documents actually contain.
      Fix: rephrase sub-questions to use document-specific vocabulary.
    - Low citation_utilization → retrieved chunks weren't used in the report.
      Cause: sub-questions retrieved off-topic chunks.
      Fix: make sub-questions more specific to the document domain.
    """
    prev_questions_block = "\n".join(
        f"  {i+1}. {q}" for i, q in enumerate(previous_sub_questions)
    )

    return f"""You are an expert research planner performing a RETRY after a failed attempt.

The previous research attempt failed quality checks. Here are the failure metrics:
  - Citation coverage: {failed_eval.citation_coverage:.2f} (target: ≥ 0.55)
    → This means {(1 - failed_eval.citation_coverage)*100:.0f}% of report sentences
      could not be traced back to source documents.
  - Citation utilization: {failed_eval.citation_utilization:.2f} (target: ≥ 0.30)
    → This means only {failed_eval.citation_utilization*100:.0f}% of retrieved chunks
      were actually used in the report.

The previous sub-questions that failed:
{prev_questions_block}

Original query: {query}

TASK: Generate new sub-questions that will retrieve MORE RELEVANT content from the documents.

Strategies to try:
1. Use more specific, document-vocabulary terms (avoid generic paraphrases)
2. Break the query into narrower factual questions
3. Include questions about specific entities, dates, or figures mentioned in the query
4. Rephrase from a different angle than the failed sub-questions above

Respond with ONLY a JSON object matching this exact schema:
{{
  "original_query": "{query}",
  "reasoning": "<explain what went wrong and your new strategy>",
  "sub_questions": [
    {{
      "question": "<new sub-question>",
      "search_queries": ["<variant 1>", "<variant 2>", "<variant 3>"]
    }}
  ]
}}

Generate exactly 3 sub_questions that are substantially different from the failed ones."""
