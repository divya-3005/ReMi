import tiktoken
from dataclasses import dataclass
from typing import List, Optional
from vectorstore.store import FaissStore
from vectorstore.retriever import search, SearchResult
from genai.prompts import QA_SYSTEM, EXPANSION_SYSTEM
from genai.client import complete

@dataclass
class QAResult:
    query: str
    answer: str
    sources: List[SearchResult]
    context_used: str

def answer(query: str, store: FaissStore, top_k: int = 5, doc_id: Optional[str] = None) -> QAResult:
    """
    Retrieves relevant chunks and generates an answer grounded in the context.
    """
    # Query Expansion: Generate keywords to improve semantic retrieval
    try:
        expanded_keywords = complete(system=EXPANSION_SYSTEM, user=query, max_tokens=50)
        search_query = f"{query} {expanded_keywords}"
    except Exception:
        search_query = query
        
    results = search(search_query, store, top_k=top_k, doc_id=doc_id)
    
    if not results:
        return QAResult(
            query=query,
            answer="not found in documents",
            sources=[],
            context_used=""
        )
        
    # Build context string
    enc = tiktoken.get_encoding("cl100k_base")
    max_tokens = 5000
    
    context_blocks = []
    current_tokens = 0
    
    for res in results:
        block = f"[source: {res.source_file}, chunk {res.chunk_index}]\n{res.chunk_text}\n"
        block_tokens = len(enc.encode(block))
        
        if current_tokens + block_tokens > max_tokens:
            break
            
        context_blocks.append(block)
        current_tokens += block_tokens
        
    context_used = "\n".join(context_blocks)
    
    # Generate answer
    user_prompt = f"Context:\n{context_used}\n\nQuestion: {query}"
    answer_text = complete(system=QA_SYSTEM, user=user_prompt, max_tokens=1024)
    
    # We only include sources that were actually used in the context string
    used_sources = results[:len(context_blocks)]
    
    return QAResult(
        query=query,
        answer=answer_text,
        sources=used_sources,
        context_used=context_used
    )
