from fastapi import APIRouter, Request, HTTPException
from api.schemas import SearchRequest, SearchResponse, SearchResultSchema
from vectorstore.retriever import search as vector_search

router = APIRouter(prefix="/search", tags=["search"])

@router.post("", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest, request: Request):
    vstore = request.app.state.vstore
    try:
        results = vector_search(req.query, vstore, top_k=req.top_k, doc_id=req.doc_id)
        out_results = []
        for r in results:
            out_results.append(SearchResultSchema(
                chunk_index=r.chunk_index,
                doc_id=r.doc_id,
                source_file=r.source_file,
                score=r.score,
                chunk_text=r.chunk_text
            ))
        return SearchResponse(results=out_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
