from fastapi import APIRouter, Request, HTTPException
from api.schemas import QARequest, QAResponse, SearchResultSchema
from genai.qa import answer as genai_ask
from grounding.linker import link
from grounding.scorer import faithfulness_score, coverage_score
from grounding.report import render
from models.research import Finding

router = APIRouter(prefix="/qa", tags=["qa"])

@router.post("", response_model=QAResponse)
async def qa_endpoint(req: QARequest, request: Request):
    vstore = request.app.state.vstore
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
        raise HTTPException(status_code=500, detail=str(e))
