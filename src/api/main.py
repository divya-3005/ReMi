"""
src/api/main.py
───────────────
FastAPI application — REST endpoints for the ReMi research pipeline.

Endpoints:
  POST /documents/upload   — ingest a PDF or TXT file into the vector store
  POST /research           — run a research query through the workflow
  GET  /documents          — list all ingested documents
  DELETE /documents/{doc_id} — remove a document from the store

All endpoints return structured JSON. Errors are returned as:
  {"detail": {"code": "ERROR_CODE", "message": "human-readable message"}}
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent.analyzer import AnalyzerAgent
from src.agent.planner import PlannerAgent
from src.agent.researcher import ResearcherAgent
from src.agent.synthesizer import SynthesizerAgent
from src.agent.workflow import ResearchWorkflow
from src.config import Settings, get_settings
from src.evaluation.evaluator import EvaluatorAgent
from src.genai.client import GeminiClient, GroqClient
from src.grounding.grounder import GrounderAgent
from src.ingestion.chunker import RecursiveChunker
from src.ingestion.errors import (
    CorruptedFileError,
    EmptyTextError,
    EncryptedFileError,
    UnsupportedFileTypeError,
)
from src.ingestion.loader import DocumentLoader
from src.models.schemas import AgentReport, DocumentMetadata
from src.vectorstore.embedder import GeminiEmbedder
from src.vectorstore.store import HybridStore

logger = logging.getLogger(__name__)

# ── Application state ─────────────────────────────────────────────────────────
# Stored as module-level singletons, initialized in the lifespan handler.
# This avoids re-constructing clients on every request.

_settings: Settings | None = None
_store: HybridStore | None = None
_workflow: ResearchWorkflow | None = None
_loader: DocumentLoader | None = None
_chunker: RecursiveChunker | None = None
_embedder: GeminiEmbedder | None = None

# In-memory document registry (doc_id → DocumentMetadata)
_document_registry: dict[str, DocumentMetadata] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services on startup."""
    global _settings, _store, _workflow, _loader, _chunker, _embedder

    _settings = get_settings()
    _settings.validate()

    # Ingestion components
    _loader = DocumentLoader()
    _chunker = RecursiveChunker(_settings)

    # Vector store
    _embedder = GeminiEmbedder(_settings)
    _store = HybridStore(_embedder, _settings)

    # Agent chain
    gemini_client = GeminiClient(_settings)
    groq_client = GroqClient(_settings)

    planner = PlannerAgent(gemini_client)
    researcher = ResearcherAgent(_store, groq_client, _settings)
    analyzer = AnalyzerAgent(groq_client, _settings)
    synthesizer = SynthesizerAgent(gemini_client)
    grounder = GrounderAgent()
    evaluator = EvaluatorAgent(_embedder)

    _workflow = ResearchWorkflow(
        planner=planner,
        researcher=researcher,
        analyzer=analyzer,
        synthesizer=synthesizer,
        grounder=grounder,
        evaluator=evaluator,
        store=_store,
        settings=_settings,
    )

    logger.info("ReMi API: all services initialized.")
    yield
    logger.info("ReMi API: shutting down.")


app = FastAPI(
    title="ReMi — Research Mind API",
    description="Upload documents and query them with full citation traceability.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": {"code": "INTERNAL_ERROR", "message": str(exc)}},
        headers={"Access-Control-Allow-Origin": "*"}
    )



# ── Request / Response schemas ────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    num_pages: int
    chunk_count: int


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    num_pages: int
    chunks_indexed: int
    message: str


class ResearchRequest(BaseModel):
    query: str


class ResearchResponse(BaseModel):
    query: str
    answer_text: str
    citations: list
    evaluation: dict
    workflow_attempts: list
    elapsed_seconds: float
    known_limitations_applied: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "docs_indexed": len(_document_registry)}


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all ingested documents with their chunk counts."""
    result = []
    for doc_id, meta in _document_registry.items():
        chunk_count = sum(
            1 for c in _store.chunks if c.doc_id == doc_id
        )
        result.append(DocumentInfo(
            doc_id=doc_id,
            filename=meta.filename,
            file_type=meta.file_type,
            num_pages=meta.num_pages,
            chunk_count=chunk_count,
        ))
    return result


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF or TXT document.

    The file is written to a temp path, processed, and then deleted.
    Chunks are embedded and indexed in the HybridStore.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(
            status_code=415,
            detail={"code": "UNSUPPORTED_FILE_TYPE", "message": f"Only .pdf and .txt are supported. Got: {suffix}"}
        )

    # Write upload to a temp file
    tmp_path = Path(f"/tmp/remi_{file.filename}")
    try:
        content = await file.read()
        tmp_path.write_bytes(content)

        try:
            meta, text = _loader.load(tmp_path)
        except UnsupportedFileTypeError as e:
            raise HTTPException(status_code=415, detail={"code": "UNSUPPORTED_FILE_TYPE", "message": str(e)})
        except EncryptedFileError as e:
            raise HTTPException(status_code=422, detail={"code": "ENCRYPTED_FILE", "message": str(e)})
        except EmptyTextError as e:
            raise HTTPException(status_code=422, detail={"code": "EMPTY_TEXT", "message": str(e)})
        except CorruptedFileError as e:
            raise HTTPException(status_code=422, detail={"code": "CORRUPTED_FILE", "message": str(e)})

        chunks = _chunker.chunk(text, meta)
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail={"code": "EMPTY_TEXT", "message": "Document produced no chunks after splitting."}
            )

        _store.add_chunks(chunks)
        _document_registry[meta.doc_id] = meta

        logger.info(f"Uploaded: {file.filename} → {len(chunks)} chunks, doc_id={meta.doc_id}")

        return UploadResponse(
            doc_id=meta.doc_id,
            filename=meta.filename,
            num_pages=meta.num_pages,
            chunks_indexed=len(chunks),
            message=f"Successfully indexed {len(chunks)} chunks from {meta.filename}.",
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    """
    Run a research query through the full multi-agent pipeline.

    Returns the grounded Markdown report, citation index, evaluation scores,
    and the workflow attempt audit trail.
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail={"code": "EMPTY_QUERY", "message": "Query must not be empty."}
        )

    if _store.count() == 0:
        raise HTTPException(
            status_code=422,
            detail={"code": "NO_DOCUMENTS", "message": "Upload at least one document before running a research query."}
        )

    report: AgentReport = _workflow.run(request.query)

    return ResearchResponse(
        query=report.query,
        answer_text=report.answer_text,
        citations=[c.model_dump() for c in report.citations],
        evaluation=report.evaluation.model_dump(),
        workflow_attempts=[a.model_dump() for a in report.workflow_attempts],
        elapsed_seconds=report.elapsed_seconds,
        known_limitations_applied=report.known_limitations_applied,
    )


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Remove a document from the registry.

    Note: This does NOT remove the document's chunks from the FAISS index
    (FAISS IndexFlatIP does not support selective deletion without rebuilding).
    The chunks remain searchable but the document no longer appears in the list.
    This is a known limitation documented in the README.
    """
    if doc_id not in _document_registry:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": f"Document {doc_id} not found."}
        )
    meta = _document_registry.pop(doc_id)
    return {"message": f"Document '{meta.filename}' removed from registry. Note: FAISS does not support in-place deletion; chunks remain in the index until the store is rebuilt."}

# Mount frontend at root. Do this last so API routes take precedence.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

