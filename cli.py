"""
cli.py
──────
ReMi command-line interface (typer).

Commands:
  remi serve          — start the FastAPI server
  remi upload <file>  — ingest a document (calls the API)
  remi ask <query>    — run a research query and print the report
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="remi",
    help="ReMi — Research Mind. Query your documents with full citation traceability.",
    add_completion=False,
)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to listen on"),
    reload: bool = typer.Option(False, help="Enable auto-reload (dev only)"),
):
    """Start the ReMi FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        typer.echo("uvicorn not found. Run: pip install uvicorn", err=True)
        raise typer.Exit(1)

    typer.echo(f"Starting ReMi API on http://{host}:{port}")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=reload)


@app.command()
def upload(
    file_path: str = typer.Argument(..., help="Path to the PDF or TXT file to upload"),
    api_url: str = typer.Option("http://localhost:8000", help="ReMi API URL"),
):
    """Upload and ingest a document."""
    import httpx

    path = Path(file_path)
    if not path.exists():
        typer.echo(f"File not found: {file_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Uploading {path.name}...")
    try:
        with httpx.Client(timeout=120) as client:
            with open(path, "rb") as f:
                response = client.post(
                    f"{api_url}/documents/upload",
                    files={"file": (path.name, f, "application/octet-stream")},
                )
        if response.status_code == 200:
            data = response.json()
            typer.echo(f"✓ {data['filename']} — {data['chunks_indexed']} chunks indexed (doc_id: {data['doc_id']})")
        else:
            typer.echo(f"✗ Upload failed ({response.status_code}): {response.json()}", err=True)
            raise typer.Exit(1)
    except httpx.ConnectError:
        typer.echo(f"Cannot connect to {api_url}. Is the server running?", err=True)
        raise typer.Exit(1)


@app.command()
def ask(
    query: str = typer.Argument(..., help="The research question"),
    api_url: str = typer.Option("http://localhost:8000", help="ReMi API URL"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON instead of formatted report"),
):
    """Run a research query and display the grounded report."""
    import httpx

    typer.echo(f"Researching: {query}\n")
    try:
        with httpx.Client(timeout=300) as client:
            response = client.post(
                f"{api_url}/research",
                json={"query": query},
            )
        if response.status_code != 200:
            typer.echo(f"✗ Research failed ({response.status_code}): {response.json()}", err=True)
            raise typer.Exit(1)

        data = response.json()

        if json_output:
            typer.echo(json.dumps(data, indent=2))
            return

        # Pretty-print the report
        typer.echo("─" * 60)
        typer.echo(data["answer_text"])
        typer.echo("\n─" * 60)

        # Evaluation scores
        ev = data["evaluation"]
        typer.echo("\nQuality Scores:")
        typer.echo(f"  citation_coverage:    {ev['citation_coverage']:.2f}")
        typer.echo(f"  citation_utilization: {ev['citation_utilization']:.2f}")
        typer.echo(f"  answer_relevance:     {ev['answer_relevance']:.2f}")
        typer.echo(f"  hallucination_risk:   {ev['hallucination_risk']:.2f}")

        # Workflow trace
        attempts = data["workflow_attempts"]
        if len(attempts) > 1:
            typer.echo(f"\nWorkflow Trace ({len(attempts)} attempts):")
            for a in attempts:
                status = "→ retry" if a["triggered_retry"] else "✓ done"
                cov = a["evaluation"]["citation_coverage"]
                util = a["evaluation"]["citation_utilization"]
                typer.echo(f"  [{a['attempt_number'] + 1}] coverage={cov:.2f}  util={util:.2f}  {status}")

        typer.echo(f"\nElapsed: {data['elapsed_seconds']:.1f}s")

    except httpx.ConnectError:
        typer.echo(f"Cannot connect to {api_url}. Is the server running?", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
