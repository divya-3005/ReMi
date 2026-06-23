import os
import json
import uuid
import glob
import dataclasses
from typing import Dict
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from api.schemas import ResearchRequest, ResearchResponse, SubQuestionSchema, EvalResultSchema
from agent.workflow import run_research

router = APIRouter(prefix="/research", tags=["research"])

def background_run_research(question: str, store, min_confidence: float, report_id: str):
    try:
        run_research(question, store, min_confidence=min_confidence, report_id=report_id)
    except Exception as e:
        print(f"Background research task failed: {e}")

@router.post("", response_model=Dict[str, str] if False else dict)
async def start_research(req: ResearchRequest, request: Request, background_tasks: BackgroundTasks):
    vstore = request.app.state.vstore
    report_id = str(uuid.uuid4())
    
    background_tasks.add_task(
        background_run_research, 
        req.question, 
        vstore, 
        req.min_confidence, 
        report_id
    )
    
    return {"report_id": report_id, "status": "started"}

@router.get("/reports", response_model=list)
async def list_reports():
    reports_dir = "data/reports"
    if not os.path.exists(reports_dir):
        return []
        
    report_files = glob.glob(os.path.join(reports_dir, "*.json"))
    results = []
    for filepath in sorted(report_files, reverse=True):
        filename = os.path.basename(filepath)
        report_id = filename.replace(".json", "")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append({
                "report_id": report_id,
                "question": data.get("research_question", "Unknown"),
                "status": data.get("status", "unknown")
            })
        except:
            pass
            
    return results

@router.get("/reports/{report_id}", response_model=ResearchResponse)
async def get_report(report_id: str):
    reports_dir = "data/reports"
    filepath = os.path.join(reports_dir, f"{report_id}.json")
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Report not found")
        
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        sqs = []
        for sq in data.get("sub_questions", []):
            sqs.append(SubQuestionSchema(
                id=sq.get("id", ""),
                question=sq.get("question", ""),
                status=sq.get("status", "pending")
            ))
            
        eval_schema = None
        if data.get("eval_result"):
            eval_schema = EvalResultSchema(**data.get("eval_result"))
            
        return ResearchResponse(
            report_id=report_id,
            question=data.get("research_question", ""),
            final_report=data.get("final_report"),
            faithfulness_score=data.get("faithfulness_score", 0.0),
            coverage_score=data.get("coverage_score", 0.0),
            sub_questions=sqs,
            status=data.get("status", "running"),
            eval_result=eval_schema
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evals", response_model=list)
async def list_evals():
    from evaluation.tracker import get_history
    history = get_history()
    return [dataclasses.asdict(e) for e in history]

@router.get("/evals/summary", response_model=dict)
async def eval_summary():
    from evaluation.tracker import get_average_scores, get_trend
    return {
        "averages": get_average_scores(),
        "trends": {
            "faithfulness": get_trend("faithfulness"),
            "answer_relevance": get_trend("answer_relevance"),
            "context_precision": get_trend("context_precision"),
            "hallucination_risk": get_trend("hallucination_risk"),
            "overall_score": get_trend("overall_score")
        }
    }
