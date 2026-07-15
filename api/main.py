import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from storage.document_store import DocumentStore
from vectorstore.store import FaissStore
from api.routes import documents, search, qa, research

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("remi_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting ResearchMind API...")
    app.state.store = DocumentStore()
    app.state.vstore = FaissStore()
    logger.info("Document and Vector stores initialized.")
    yield
    # Shutdown
    logger.info("Shutting down ResearchMind API...")

app = FastAPI(
    title="ResearchMind API",
    description="Backend API for the ResearchMind multi-agent RAG system.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow any origin (Vercel, Localhost, etc)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred. Please try again later."}
    )

app.include_router(documents.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(qa.router, prefix="/api")
app.include_router(research.router, prefix="/api")

@app.get("/api/health", tags=["system"])
async def health_check():
    return {"status": "ok", "version": "1.0.0"}
