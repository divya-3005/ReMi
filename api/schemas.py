from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class IngestResponse(BaseModel):
    doc_id: str = Field(..., description="Unique UUID for the ingested document")
    filename: str = Field(..., description="Original name of the file")
    chunk_count: int = Field(..., description="Number of searchable text chunks created")
    status: str = Field(..., description="Success or failure status")

class SearchRequest(BaseModel):
    query: str = Field(..., description="The natural language search query", min_length=1)
    top_k: int = Field(5, description="Number of results to return", ge=1, le=20)
    doc_id: Optional[str] = Field(None, description="Optional UUID to restrict search to a specific document")

class SearchResultSchema(BaseModel):
    chunk_index: int
    doc_id: str
    source_file: str
    score: float
    chunk_text: str

class SearchResponse(BaseModel):
    results: List[SearchResultSchema]

class QARequest(BaseModel):
    query: str = Field(..., description="The question to ask", min_length=1)
    top_k: int = Field(5, description="Number of relevant chunks to retrieve for context", ge=1, le=20)
    doc_id: Optional[str] = Field(None, description="Optional UUID to restrict Q&A to a specific document")

class QAResponse(BaseModel):
    answer: str = Field(..., description="The generated answer with markdown citations")
    sources: List[SearchResultSchema] = Field(..., description="List of source chunks used to generate the answer")
    faithfulness_score: float = Field(..., description="Score 0-1 representing how well the answer is grounded in sources")
    coverage_score: float = Field(..., description="Score 0-1 representing how well the answer addresses the question")

class ResearchRequest(BaseModel):
    question: str = Field(..., description="Complex research question to investigate", min_length=1)
    min_confidence: float = Field(0.3, description="Minimum confidence threshold for findings", ge=0.0, le=1.0)

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
