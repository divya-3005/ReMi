from datetime import datetime, timezone
from models.research import ResearchReport, EvalResult
from evaluation.metrics import faithfulness, answer_relevance, context_precision, hallucination_risk

def evaluate(report: ResearchReport, report_id: str) -> EvalResult:
    f_score = 0.0
    a_score = 0.0
    c_score = 0.0
    h_score = 0.0
    
    # Evaluate Faithfulness & Hallucination Risk
    if report.linked_report:
        try:
            f_score = faithfulness(report.linked_report)
        except Exception as e:
            print(f"Warning: faithfulness metric failed: {e}")
        try:
            h_score = hallucination_risk(report.linked_report)
        except Exception as e:
            print(f"Warning: hallucination_risk metric failed: {e}")
            
    # Evaluate Answer Relevance
    if report.final_report:
        try:
            a_score = answer_relevance(report.research_question, report.final_report)
        except Exception as e:
            print(f"Warning: answer_relevance metric failed: {e}")
            
    # Evaluate Context Precision
    try:
        chunks = []
        for finding in report.findings:
            for src in finding.sources:
                chunks.append(src.text)
        if chunks:
            c_score = context_precision(report.research_question, chunks)
    except Exception as e:
        print(f"Warning: context_precision metric failed: {e}")
        
    # Calculate Overall Score
    overall_score = (f_score * 0.35) + (a_score * 0.25) + (c_score * 0.25) + ((1.0 - h_score) * 0.15)
    
    return EvalResult(
        report_id=report_id,
        question=report.research_question,
        faithfulness=f_score,
        answer_relevance=a_score,
        context_precision=c_score,
        hallucination_risk=h_score,
        overall_score=overall_score,
        created_at=datetime.now(timezone.utc).isoformat()
    )
