import logging
from fastapi import APIRouter, Request, HTTPException
from api.schemas import QARequest, QAResponse, SearchResultSchema
from genai.qa import answer as genai_ask
from grounding.linker import link
from grounding.scorer import faithfulness_score, coverage_score
from grounding.report import render
from models.research import Finding

router = APIRouter(prefix="/qa", tags=["qa"])
logger = logging.getLogger("remi_api.qa")

@router.post("", response_model=QAResponse)
async def qa_endpoint(req: QARequest, request: Request):
    vstore = request.app.state.vstore
    
    if len(vstore.metadata) == 0:
        raise HTTPException(status_code=400, detail="No documents have been indexed yet. Please upload and index a document first.")
        
    try:
        result = genai_ask(req.query, vstore, top_k=req.top_k, doc_id=req.doc_id)
        
        finding = Finding(sub_question_id="ask", answer=result.answer, sources=result.sources, confidence_score=1.0)
        linked = link(result.answer, [finding])
        grounded_md = render(linked)
        faith = faithfulness_score(linked)
        cov = coverage_score(req.query, linked)
        
        out_sources = []
        for r in result.sources:
            out_sources.append(SearchResultSchema(
                chunk_index=r.chunk_index,
                doc_id=r.doc_id,
                source_file=r.source_file,
                score=r.score,
                chunk_text=r.chunk_text
            ))
            
        return QAResponse(
            answer=grounded_md,
            sources=out_sources,
            faithfulness_score=faith,
            coverage_score=cov
        )
    except Exception as e:
        logger.error(f"Error generating QA response for query '{req.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")
