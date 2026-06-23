from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    status: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    doc_id: Optional[str] = None

class SearchResultSchema(BaseModel):
    chunk_index: int
    doc_id: str
    source_file: str
    score: float
    chunk_text: str

class SearchResponse(BaseModel):
    results: List[SearchResultSchema]

class QARequest(BaseModel):
    query: str
    top_k: int = 5
    doc_id: Optional[str] = None

class QAResponse(BaseModel):
    answer: str
    sources: List[SearchResultSchema]
    faithfulness_score: float
    coverage_score: float

class ResearchRequest(BaseModel):
    question: str
    min_confidence: float = 0.3

class SubQuestionSchema(BaseModel):
    id: str
    question: str
    status: str

class EvalResultSchema(BaseModel):
    report_id: str
    question: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    hallucination_risk: float
    overall_score: float
    created_at: str

class ResearchResponse(BaseModel):
    report_id: str
    question: str
    final_report: Optional[str]
    faithfulness_score: float
    coverage_score: float
    sub_questions: List[SubQuestionSchema]
    status: str
    eval_result: Optional[EvalResultSchema] = None
