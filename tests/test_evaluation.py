import os
import json
import pytest
from evaluation.metrics import faithfulness, hallucination_risk, context_precision
from evaluation.tracker import append_eval, get_history, get_average_scores
from models.research import LinkedAnswer, LinkedSentence, EvidenceSpan, EvalResult

def test_faithfulness_and_hallucination():
    # Fully grounded
    la = LinkedAnswer("ans", [LinkedSentence("s1", [], True)])
    assert faithfulness(la) == 1.0
    assert hallucination_risk(la) == 0.0

    # Partially grounded
    la = LinkedAnswer("ans", [LinkedSentence("s1", [], True), LinkedSentence("s2", [], False)])
    assert faithfulness(la) == 0.5
    assert hallucination_risk(la) == 0.5

    # Completely hallucinated
    la = LinkedAnswer("ans", [LinkedSentence("s1", [], False)])
    assert faithfulness(la) == 0.0
    assert hallucination_risk(la) == 1.0

def test_context_precision_empty():
    assert context_precision("question", []) == 0.0

def test_tracker_io(tmp_path, monkeypatch):
    test_file = tmp_path / "evals.json"
    monkeypatch.setattr("evaluation.tracker.EVALS_FILE", str(test_file))
    
    er = EvalResult(
        report_id="test1",
        question="q",
        faithfulness=0.8,
        answer_relevance=0.9,
        context_precision=0.7,
        hallucination_risk=0.2,
        overall_score=0.8,
        created_at="2023-01-01T00:00:00"
    )
    
    append_eval(er)
    history = get_history()
    assert len(history) == 1
    assert history[0].report_id == "test1"
    
    avg = get_average_scores()
    assert avg["faithfulness"] == 0.8
