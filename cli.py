import os
import uuid
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from ingestion.loader import load_file
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_document
from storage.document_store import DocumentStore
from models.document import Document
from vectorstore.store import FaissStore
from vectorstore.retriever import search as vector_search
from genai.qa import answer as genai_ask
from genai.summarizer import summarize_document
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.markdown import Markdown

app = typer.Typer(help="ResearchMind Ingestion Pipeline CLI")
console = Console()
store = DocumentStore()
vstore = FaissStore()

@app.command()
def ingest(path: str = typer.Argument(..., help="Path to a PDF/TXT file or folder containing them")):
    """Ingests a file or all PDF/TXT files in a folder, cleans, chunks, and stores them."""
    if not os.path.exists(path):
        console.print(f"[bold red]Error:[/] Path does not exist: {path}")
        raise typer.Exit(code=1)

    files_to_process = []
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith((".pdf", ".txt")):
                    files_to_process.append(os.path.join(root, file))
    else:
        if path.lower().endswith((".pdf", ".txt")):
            files_to_process.append(path)
        else:
            console.print(
                f"[bold red]Error:[/] Unsupported file extension for path: {path}. Only .pdf and .txt are supported."
            )
            raise typer.Exit(code=1)

    if not files_to_process:
        console.print("[bold yellow]No PDF or TXT files found to process.[/]")
        return

    table = Table(title="Ingestion Summary")
    table.add_column("Filename", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Size (Bytes)", style="magenta")
    table.add_column("Pages", style="blue")
    table.add_column("Chunks", style="yellow")
    table.add_column("Document ID", style="dim")

    for file_path in files_to_process:
        filename = os.path.basename(file_path)
        try:
            with console.status(f"[bold green]Processing {filename}...[/]"):
                # 1. Load file
                raw_text, metadata = load_file(file_path)

                # 2. Clean text
                cleaned = clean_text(raw_text)

                # 3. Create document ID
                doc_id = str(uuid.uuid4())

                # 4. Chunk document
                chunks = chunk_document(
                    document_id=doc_id,
                    cleaned_text=cleaned,
                    source_file=filename
                )

                # 5. Save to store
                doc = Document(
                    id=doc_id,
                    filename=filename,
                    raw_text=raw_text,
                    file_size_bytes=metadata["file_size_bytes"],
                    ingested_at=metadata["ingested_at"],
                    page_count=metadata["page_count"],
                    metadata={"cleaned_text_length": len(cleaned)}
                )
                store.save(doc, chunks)

            table.add_row(
                filename,
                "Success",
                f"{doc.file_size_bytes:,}",
                str(doc.page_count or "N/A"),
                str(len(chunks)),
                doc.id
            )
        except Exception as e:
            table.add_row(filename, f"[bold red]Failed: {str(e)}[/]", "N/A", "N/A", "N/A", "N/A")

    console.print(table)

@app.command(name="list")
def list_documents():
    """Lists all ingested documents in the store."""
    docs = store.list_all()
    if not docs:
        console.print("[bold yellow]No documents ingested yet. Use the 'ingest' command.[/]")
        return

    table = Table(title="Ingested Documents")
    table.add_column("Document ID", style="dim")
    table.add_column("Filename", style="cyan")
    table.add_column("Size (Bytes)", style="magenta")
    table.add_column("Pages", style="blue")
    table.add_column("Chunks", style="yellow")
    table.add_column("Ingested At", style="green")

    for doc in docs:
        chunk_count = store.get_chunk_count(doc.id)
        table.add_row(
            doc.id,
            doc.filename,
            f"{doc.file_size_bytes:,}",
            str(doc.page_count or "N/A"),
            str(chunk_count),
            doc.ingested_at
        )

    console.print(table)

@app.command()
def show(doc_id: str = typer.Argument(..., help="The UUID of the document to inspect")):
    """Shows the metadata and first 3 chunks of an ingested document."""
    res = store.get_by_id(doc_id)
    if not res:
        console.print(f"[bold red]Error:[/] Document with ID {doc_id} not found.")
        raise typer.Exit(code=1)

    doc, chunks = res

    # Print Document metadata
    console.print(
        Panel(
            f"[bold]Filename:[/] {doc.filename}\n"
            f"[bold]Size:[/] {doc.file_size_bytes:,} bytes\n"
            f"[bold]Pages:[/] {doc.page_count or 'N/A'}\n"
            f"[bold]Total Chunks:[/] {len(chunks)}\n"
            f"[bold]Ingested At:[/] {doc.ingested_at}",
            title=f"Document Metadata - {doc.id}",
            border_style="cyan"
        )
    )

    # Show first 3 chunks
    chunks_to_show = chunks[:3]
    for idx, chunk in enumerate(chunks_to_show):
        console.print(
            Panel(
                chunk.content,
                title=(
                    f"Chunk {idx+1}/{len(chunks)} (ID: {chunk.chunk_id}) - "
                    f"Page {chunk.page_number or 'N/A'} "
                    f"(Char Offset: {chunk.char_offset}, Tokens: {chunk.token_count})"
                ),
                border_style="yellow"
            )
        )

    if len(chunks) > 3:
        console.print(f"[dim]... and {len(chunks) - 3} more chunks.[/]")

@app.command()
def delete(doc_id: str = typer.Argument(..., help="The UUID of the document to delete")):
    """Deletes a document from the store."""
    success = store.delete(doc_id)
    if success:
        vstore.remove_document(doc_id)
        console.print(f"[bold green]Success:[/] Document {doc_id} deleted from JSON and vector store.")
    else:
        console.print(f"[bold red]Error:[/] Document {doc_id} not found.")

@app.command()
def analyze(doc_id: str = typer.Argument(..., help="The UUID of the document to analyze")):
    """Runs NLP analysis on a document and displays the results."""
    from nlp.analyzer import analyze_document
    
    with console.status(f"[bold green]Analyzing document {doc_id}...[/]"):
        success = analyze_document(doc_id, store)
        
    if not success:
        console.print(f"[bold red]Error:[/] Document {doc_id} not found.")
        raise typer.Exit(code=1)
        
    res = store.get_by_id(doc_id)
    if not res:
        return
    doc, _ = res
    nlp_data = doc.nlp
    
    if not nlp_data:
        console.print(f"[bold yellow]No NLP data generated for document {doc_id}.[/]")
        return
        
    # 1. Top 10 Keywords
    kw_table = Table(title="Top 10 Keywords (Document Level)")
    kw_table.add_column("Keyword", style="cyan")
    kw_table.add_column("TF-IDF Score", style="magenta")
    
    for kw in nlp_data.get("keywords", {}).get("document_level", [])[:10]:
        kw_table.add_row(kw["keyword"], f"{kw['score']:.4f}")
        
    console.print(kw_table)
    
    # 2. Top 10 Entities grouped by type
    ent_table = Table(title="Top 10 Entities")
    ent_table.add_column("Entity Text", style="cyan")
    ent_table.add_column("Type", style="yellow")
    ent_table.add_column("Frequency", style="green")
    
    doc_ents = nlp_data.get("entities", {}).get("document_level", [])
    for ent in doc_ents[:10]:
        ent_table.add_row(ent["text"], ent["label"], str(ent["count"]))
        
    console.print(ent_table)
    
    # 3. 3 Most important sentences
    all_sents = []
    for chunk_sents in nlp_data.get("importance", {}).get("per_chunk", {}).values():
        all_sents.extend(chunk_sents)
        
    all_sents.sort(key=lambda x: x["score"], reverse=True)
    
    console.print(Panel("[bold]Top 3 Important Sentences[/]"))
    for i, sent_data in enumerate(all_sents[:3]):
        console.print(f"[bold yellow]{i+1}.[/] {sent_data['sentence']} [dim](Score: {sent_data['score']:.4f})[/]")

@app.command()
def index(doc_id: str = typer.Argument(..., help="The UUID of the document to index in FAISS")):
    """Embeds and indexes a document's chunks in the vector store."""
    res = store.get_by_id(doc_id)
    if not res:
        console.print(f"[bold red]Error:[/] Document {doc_id} not found.")
        raise typer.Exit(code=1)
        
    doc, chunks = res
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Indexing document {doc_id}...", total=len(chunks))
        
        # We can just add all chunks at once, embedder handles batching
        # But progress bar is nicer if we add them in one go anyway, since embed_texts handles the batch
        # To show real progress we could chunk the chunks, but for now we just show a spinner-like progress
        # Since add_document does it all at once, we'll just step the progress bar to 100% after
        
    with console.status(f"[bold green]Embedding and indexing {len(chunks)} chunks...[/]"):
        vstore.add_document(doc_id, chunks, doc.filename)
        
    console.print(f"[bold green]Success:[/] Indexed document {doc_id} ({len(chunks)} chunks).")

@app.command("index-all")
def index_all():
    """Indexes all unindexed documents in the vector store."""
    docs = store.list_all()
    if not docs:
        console.print("[bold yellow]No documents in store.[/]")
        return
        
    # Find which docs are already indexed
    indexed_doc_ids = {meta.get("doc_id") for meta in vstore.metadata}
    docs_to_index = [d for d in docs if d.id not in indexed_doc_ids]
    
    if not docs_to_index:
        console.print("[bold green]All documents are already indexed![/]")
        return
        
    console.print(f"[bold]Found {len(docs_to_index)} documents to index.[/]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Indexing documents...", total=len(docs_to_index))
        for d in docs_to_index:
            res = store.get_by_id(d.id)
            if res:
                _, chunks = res
                vstore.add_document(d.id, chunks, d.filename)
            progress.advance(task)
            
    console.print("[bold green]Success:[/] Finished indexing all documents.")

@app.command()
def search(query: str = typer.Argument(..., help="The search query")):
    """Searches the vector store for relevant chunks."""
    with console.status(f"[bold green]Searching for '{query}'...[/]"):
        results = vector_search(query, vstore, top_k=5)
        
    if not results:
        console.print("[bold yellow]No results found or vector store is empty.[/]")
        return
        
    table = Table(title=f"Semantic Search Results: '{query}'")
    table.add_column("Rank", style="cyan", justify="right")
    table.add_column("Score", style="magenta")
    table.add_column("Source", style="green")
    table.add_column("Excerpt", style="white")
    
    for i, res in enumerate(results):
        # Format excerpt: replace newlines with spaces, truncate
        excerpt = res.chunk_text.replace("\n", " ")
        if len(excerpt) > 100:
            excerpt = excerpt[:97] + "..."
            
        table.add_row(
            str(i+1),
            f"{res.score:.4f}",
            res.source_file,
            excerpt
        )
        
    console.print(table)

@app.command()
def ask(query: str = typer.Argument(..., help="The search query or question"), doc_id: str = typer.Option(None, "--doc", help="Optional document ID to scope the search")):
    """Asks a question using the retrieved context from the vector store."""
    with console.status(f"[bold green]Generating answer for '{query}'...[/]"):
        try:
            result = genai_ask(query, vstore, top_k=5, doc_id=doc_id)
        except Exception as e:
            console.print(f"[bold red]Error:[/] {str(e)}")
            raise typer.Exit(code=1)
            
    console.print(Panel(Markdown(result.answer), title="Answer", border_style="green"))
    
    if result.sources:
        table = Table(title="Sources Cited")
        table.add_column("Source File", style="cyan")
        table.add_column("Chunk", style="yellow")
        table.add_column("Relevance", style="magenta")
        
        for src in result.sources:
            table.add_row(src.source_file, str(src.chunk_index), f"{src.score:.4f}")
            
        console.print(table)
    else:
        console.print("[yellow]No sources were found to ground this answer.[/]")

@app.command()
def summarize(doc_id: str = typer.Argument(..., help="The UUID of the document to summarize")):
    """Generates a hierarchical summary of a document."""
    with console.status(f"[bold green]Generating hierarchical summary for {doc_id}...[/]"):
        try:
            summary = summarize_document(doc_id, store)
        except Exception as e:
            console.print(f"[bold red]Error:[/] {str(e)}")
            raise typer.Exit(code=1)
            
    console.print(Panel(Markdown(summary.full_summary), title=f"Summary: {summary.source_file}", border_style="blue"))

if __name__ == "__main__":
    app()
