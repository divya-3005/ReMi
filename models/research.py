from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from vectorstore.retriever import SearchResult

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
class ResearchReport:
    research_question: str
    sub_questions: List[SubQuestion]
    findings: List[Finding]
    final_report: str
    created_at: str
