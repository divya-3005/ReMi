from models.research import SubQuestion, Finding
from genai.qa import answer
from vectorstore.store import FaissStore

def research_subquestion(sq: SubQuestion, store: FaissStore, top_k: int = 5) -> Finding:
    """
    Runs retrieval-augmented generation for a single sub-question.
    """
    qa_result = answer(query=sq.question, store=store, top_k=top_k)
    
    confidence_score = 0.0
    if qa_result.sources:
        # Calculate average semantic relevance of retrieved chunks
        scores = [src.score for src in qa_result.sources]
        confidence_score = sum(scores) / len(scores)
        
    # If no sources found or confidence is 0
    if not qa_result.sources or confidence_score == 0.0:
        sq.status = "failed"
    else:
        sq.status = "done"
        
    return Finding(
        sub_question_id=sq.id,
        answer=qa_result.answer,
        sources=qa_result.sources,
        confidence_score=confidence_score
    )
