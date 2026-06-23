import os
import json
import dataclasses
from typing import List, Dict
from models.research import EvalResult

EVALS_FILE = "data/evals.json"

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

def append_eval(result: EvalResult):
    os.makedirs(os.path.dirname(EVALS_FILE), exist_ok=True)
    
    history = get_history()
    # Check if report_id already exists to avoid duplicates
    existing = [r for r in history if r.report_id == result.report_id]
    if not existing:
        history.append(result)
        with open(EVALS_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, cls=EnhancedJSONEncoder, indent=2)

def get_history() -> List[EvalResult]:
    if not os.path.exists(EVALS_FILE):
        return []
    try:
        with open(EVALS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [EvalResult(**item) for item in data]
    except Exception as e:
        print(f"Failed to read history: {e}")
        return []

def get_average_scores() -> Dict[str, float]:
    history = get_history()
    if not history:
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_precision": 0.0,
            "hallucination_risk": 0.0,
            "overall_score": 0.0
        }
        
    def avg(metric: str) -> float:
        return sum(getattr(r, metric) for r in history) / len(history)
        
    return {
        "faithfulness": avg("faithfulness"),
        "answer_relevance": avg("answer_relevance"),
        "context_precision": avg("context_precision"),
        "hallucination_risk": avg("hallucination_risk"),
        "overall_score": avg("overall_score")
    }

def get_trend(metric: str) -> List[float]:
    history = get_history()
    return [getattr(r, metric) for r in history]
