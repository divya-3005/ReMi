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

console = Console()

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

def run_research(question: str, store: FaissStore, min_confidence: float = 0.3) -> ResearchReport:
    """
    Orchestrates the full research pipeline.
    """
    console.print("[bold cyan]Step 1: Planning...[/]")
    try:
        sub_questions = plan(question)
    except Exception as e:
        console.print(f"[bold red]Planning failed:[/] {str(e)}")
        raise

    findings = []
    
    console.print(f"[bold cyan]Step 2: Researching {len(sub_questions)} sub-questions...[/]")
    for i, sq in enumerate(sub_questions):
        console.print(f"  [yellow]Researching sub-question {i+1}/{len(sub_questions)}:[/] {sq.question}")
        try:
            finding = research_subquestion(sq, store=store, top_k=5)
            findings.append(finding)
        except Exception as e:
            console.print(f"  [red]Failed to research sub-question {i+1}:[/] {str(e)}")
            sq.status = "failed"
            continue

    console.print("[bold cyan]Step 3: Analyzing evidence...[/]")
    cleaned_findings = analyze_findings(findings, min_confidence=min_confidence)
    console.print(f"  [green]Retained {len(cleaned_findings)} high-quality findings.[/]")

    console.print("[bold cyan]Step 4: Synthesizing report...[/]")
    try:
        final_report = synthesize(question, cleaned_findings)
    except Exception as e:
        console.print(f"[bold red]Synthesis failed:[/] {str(e)}")
        raise

    report = ResearchReport(
        research_question=question,
        sub_questions=sub_questions,
        findings=cleaned_findings,
        final_report=final_report,
        created_at=datetime.now(timezone.utc).isoformat()
    )

    # Save to JSON
    reports_dir = "data/reports"
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(reports_dir, f"{timestamp}_report.json")
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, cls=EnhancedJSONEncoder, indent=2)
        
    console.print(f"[bold green]Report saved to:[/] {filepath}")
    
    return report
