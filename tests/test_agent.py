import pytest
from unittest.mock import patch, MagicMock
from models.research import SubQuestion, Finding
from agent.planner import plan
from agent.researcher import research_subquestion
from agent.analyzer import analyze_findings
from agent.synthesizer import synthesize
from agent.workflow import run_research
from vectorstore.retriever import SearchResult
from genai.qa import QAResult
import os

@patch("agent.planner.complete")
def test_planner(mock_complete):
    mock_complete.return_value = "1. First Q?\n2. Second Q?\n3. Third Q?"
    sqs = plan("What is AI?")
    
    assert len(sqs) == 3
    assert sqs[0].question == "First Q?"
    assert sqs[1].question == "Second Q?"
    assert sqs[2].question == "Third Q?"
    assert sqs[0].status == "pending"

@patch("agent.researcher.answer")
def test_researcher_success(mock_answer):
    mock_answer.return_value = QAResult(
        query="First Q?",
        answer="The answer",
        sources=[SearchResult("c1", "d1", "file.txt", 0.8, "text", 1)],
        context_used="text"
    )
    
    sq = SubQuestion("id1", "First Q?")
    finding = research_subquestion(sq, None)
    
    assert sq.status == "done"
    assert finding.answer == "The answer"
    assert finding.confidence_score == 0.8
    
@patch("agent.researcher.answer")
def test_researcher_failed(mock_answer):
    mock_answer.return_value = QAResult(
        query="Q?",
        answer="I don't know",
        sources=[],
        context_used=""
    )
    
    sq = SubQuestion("id2", "Q?")
    finding = research_subquestion(sq, None)
    
    assert sq.status == "failed"
    assert finding.confidence_score == 0.0

def test_analyzer():
    src = SearchResult("c1", "d1", "file.txt", 0.9, "text", 1)
    src_dup = SearchResult("c1", "d1", "file.txt", 0.9, "text", 1) # exact duplicate
    
    f1 = Finding("sq1", "ans1", [src], 0.1) # low confidence
    f2 = Finding("sq2", "ans2", [src, src_dup], 0.9) # high confidence with dup
    
    cleaned = analyze_findings([f1, f2], min_confidence=0.3)
    
    assert len(cleaned) == 1
    assert cleaned[0].confidence_score == 0.9
    assert len(cleaned[0].sources) == 1 # dedup'ed

@patch("agent.synthesizer.complete")
def test_synthesizer(mock_complete):
    mock_complete.return_value = "The report content"
    
    src = SearchResult("c1", "d1", "file.txt", 0.9, "text", 1)
    f1 = Finding("sq1", "ans1", [src], 0.9)
    
    report = synthesize("Question", [f1])
    
    assert "The report content" in report
    assert "file.txt" in report

@patch("agent.workflow.synthesize")
@patch("agent.workflow.analyze_findings")
@patch("agent.workflow.research_subquestion")
@patch("agent.workflow.plan")
def test_workflow(mock_plan, mock_research, mock_analyze, mock_synthesize, tmpdir):
    mock_plan.return_value = [SubQuestion("1", "sq1")]
    f1 = Finding("1", "ans1", [], 0.9)
    mock_research.return_value = f1
    mock_analyze.return_value = [f1]
    mock_synthesize.return_value = "final report"
    
    # Run the workflow
    with patch("os.makedirs"):
        with patch("builtins.open", MagicMock()):
            report = run_research("question", None, min_confidence=0.3)
            
    assert report.research_question == "question"
    assert report.final_report == "final report"
    assert mock_plan.call_count == 1
    assert mock_research.call_count == 1
    assert mock_analyze.call_count == 1
    assert mock_synthesize.call_count == 1
