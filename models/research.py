from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from vectorstore.retriever import SearchResult

@dataclass
class EvidenceSpan:
    chunk_index: int
    source_file: str
    start_char: int
    end_char: int
    span_text: str
    relevance_score: float

@dataclass
class LinkedSentence:
    sentence: str
    evidence: List[EvidenceSpan]
    grounded: bool

@dataclass
class LinkedAnswer:
    answer: str
    sentences: List[LinkedSentence]

@dataclass
class SubQuestion:
    id: str
    question: str
    status: str = "pending"  # pending, done, failed

@dataclass
class Finding:
    sub_question_id: str
    answer: str
    sources: List[SearchResult]
    confidence_score: float

@dataclass
class EvalResult:
    report_id: str
    question: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    hallucination_risk: float
    overall_score: float
    created_at: str

@dataclass
class ResearchReport:
    research_question: str
    sub_questions: List[SubQuestion]
    findings: List[Finding]
    final_report: str
    created_at: str
    linked_report: Optional[LinkedAnswer] = None
    faithfulness_score: float = 0.0
    coverage_score: float = 0.0
    status: str = "running"
    eval_result: Optional[EvalResult] = None
