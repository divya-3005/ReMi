from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from storage.document_store import DocumentStore
from vectorstore.store import FaissStore
from api.routes import documents, search, qa, research

app = FastAPI(title="ResearchMind API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Load once and attach to app state
    app.state.store = DocumentStore()
    app.state.vstore = FaissStore()

app.include_router(documents.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(qa.router, prefix="/api")
app.include_router(research.router, prefix="/api")

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}
