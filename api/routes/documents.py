import os
import uuid
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from api.schemas import IngestResponse
from ingestion.loader import load_file
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_document
from models.document import Document

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: Request, file: UploadFile = File(...)):
    store = request.app.state.store
    vstore = request.app.state.vstore

    temp_path = f"/tmp/{file.filename}"
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
            
        raw_text, metadata = load_file(temp_path)
        cleaned = clean_text(raw_text)
        
        doc_id = str(uuid.uuid4())
        chunks = chunk_document(
            document_id=doc_id,
            cleaned_text=cleaned,
            source_file=file.filename
        )
        
        doc = Document(
            id=doc_id,
            filename=file.filename,
            raw_text=raw_text,
            file_size_bytes=metadata["file_size_bytes"],
            ingested_at=metadata["ingested_at"],
            page_count=metadata["page_count"],
            metadata={"cleaned_text_length": len(cleaned)}
        )
        
        store.save(doc, chunks)
        vstore.add_document(doc_id, chunks, doc.filename)
        
        return IngestResponse(
            doc_id=doc_id,
            filename=file.filename,
            chunk_count=len(chunks),
            status="Success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.get("", response_model=List[Dict[str, Any]])
async def list_documents(request: Request):
    store = request.app.state.store
    docs = store.list_all()
    results = []
    for doc in docs:
        chunk_count = store.get_chunk_count(doc.id)
        results.append({
            "id": doc.id,
            "filename": doc.filename,
            "file_size_bytes": doc.file_size_bytes,
            "page_count": doc.page_count,
            "chunk_count": chunk_count,
            "ingested_at": doc.ingested_at
        })
    return results

@router.delete("/{doc_id}")
async def delete_document(doc_id: str, request: Request):
    store = request.app.state.store
    vstore = request.app.state.vstore
    
    success = store.delete(doc_id)
    if success:
        vstore.remove_document(doc_id)
        return {"status": "success", "message": f"Deleted {doc_id}"}
    else:
        raise HTTPException(status_code=404, detail="Document not found")
