import os
import json
import dataclasses
from datetime import datetime, timezone
from rich.console import Console
from models.research import ResearchReport, SubQuestion, Finding
from agent.planner import plan
from agent.researcher import research_subquestion
from agent.analyzer import analyze_findings
from agent.synthesizer import synthesize
from vectorstore.store import FaissStore
from grounding.linker import link
from grounding.scorer import faithfulness_score, coverage_score
from grounding.report import render

console = Console()

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

def run_research(question: str, store: FaissStore, min_confidence: float = 0.01, report_id: str = None) -> ResearchReport:
    """
    Orchestrates the full research pipeline.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if report_id is None:
        report_id = f"{timestamp}_report"
        
    reports_dir = "data/reports"
    os.makedirs(reports_dir, exist_ok=True)
    filepath = os.path.join(reports_dir, f"{report_id}.json")
    
    report = ResearchReport(
        research_question=question,
        sub_questions=[],
        findings=[],
        final_report=None,
        created_at=datetime.now(timezone.utc).isoformat(),
        status="running"
    )
    
    def flush_state():
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, cls=EnhancedJSONEncoder, indent=2)

    flush_state()

    console.print("[bold cyan]Step 1: Planning...[/]")
    try:
        sub_questions = plan(question)
        report.sub_questions = sub_questions
        flush_state()
    except Exception as e:
        console.print(f"[bold red]Planning failed:[/] {str(e)}")
        report.status = "failed"
        flush_state()
        raise

    findings = []
    import concurrent.futures
    
    console.print(f"[bold cyan]Step 2: Researching {len(sub_questions)} sub-questions in parallel...[/]")
    
    # Mark all as running initially
    for sq in sub_questions:
        sq.status = "running"
    flush_state()

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_idx = {
            executor.submit(research_subquestion, sq, store, 10): (i, sq)
            for i, sq in enumerate(sub_questions)
        }
        
        for future in concurrent.futures.as_completed(future_to_idx):
            i, sq = future_to_idx[future]
            try:
                finding = future.result()
                findings.append(finding)
                report.findings = findings
                console.print(f"  [green]Completed sub-question {i+1}:[/] {sq.question}")
                flush_state()
            except Exception as e:
                console.print(f"  [red]Failed to research sub-question {i+1}:[/] {str(e)}")
                sq.status = "failed"
                flush_state()

    console.print("[bold cyan]Step 3: Analyzing evidence...[/]")
    cleaned_findings = analyze_findings(findings, min_confidence=min_confidence)
    report.findings = cleaned_findings
    flush_state()
    console.print(f"  [green]Retained {len(cleaned_findings)} high-quality findings.[/]")

    console.print("[bold cyan]Step 4: Synthesizing report...[/]")
    try:
        final_report = synthesize(question, cleaned_findings)
        report.final_report = final_report
        flush_state()
    except Exception as e:
        console.print(f"[bold red]Synthesis failed:[/] {str(e)}")
        report.status = "failed"
        flush_state()
        raise

    console.print("[bold cyan]Step 5: Grounding evidence...[/]")
    try:
        linked_report = link(final_report, cleaned_findings)
        faith = faithfulness_score(linked_report)
        cov = coverage_score(question, linked_report)
        grounded_md = render(linked_report)
        
        # Override the final report with the grounded version
        report.final_report = grounded_md
        report.linked_report = linked_report
        report.faithfulness_score = faith
        report.coverage_score = cov
        console.print(f"  [green]Grounding complete. Faithfulness: {faith:.2f} | Coverage: {cov:.2f}[/]")
    except Exception as e:
        console.print(f"  [yellow]Grounding failed, falling back to ungrounded report:[/] {str(e)}")

    console.print("[bold cyan]Step 6: Running evaluation metrics...[/]")
    try:
        from evaluation.evaluator import evaluate
        from evaluation.tracker import append_eval
        
        eval_result = evaluate(report, report_id=report_id)
        report.eval_result = eval_result
        append_eval(eval_result)
        console.print(f"  [green]Evaluation complete. Overall score: {eval_result.overall_score:.2f}[/]")
    except Exception as e:
        console.print(f"  [yellow]Evaluation failed:[/] {str(e)}")

    report.status = "complete"
    flush_state()
        
    console.print(f"[bold green]Report saved to:[/] {filepath}")
    
    return report
