from dataclasses import dataclass
from typing import List, Optional
from vectorstore.store import FaissStore
from vectorstore.embedder import embed_query

@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    source_file: str
    score: float
    chunk_text: str
    chunk_index: int

def search(query: str, store: FaissStore, top_k: int = 5, doc_id: Optional[str] = None, cross_doc: bool = True) -> List[SearchResult]:
    """
    Searches the vector store for the most relevant chunks.
    
    Args:
        query: The search string.
        store: FaissStore instance.
        top_k: Max results to return.
        doc_id: Optional document ID to restrict search.
        cross_doc: If False, deduplicates results so only 1 chunk per document is returned.
        
    Returns:
        List of SearchResult.
    """
    if store.total_chunks() == 0:
        import warnings
        warnings.warn("Vector store is empty.")
        return []
        
    # Embed query
    query_emb = embed_query(query)
    
    # We query more than top_k because we might filter by doc_id or deduplicate
    search_k = min(store.total_chunks(), max(top_k * 10, 50))
    
    # FAISS search returns scores and indices
    scores, indices = store.index.search(query_emb, search_k)
    
    results = []
    seen_docs = set()
    
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
            
        meta = store.metadata[idx]
        
        # Filter by doc_id
        if doc_id and meta.get("doc_id") != doc_id:
            continue
            
        # Deduplicate
        current_doc = meta.get("doc_id")
        if not cross_doc and current_doc in seen_docs:
            continue
            
        seen_docs.add(current_doc)
        
        results.append(SearchResult(
            chunk_id=meta.get("chunk_id"),
            doc_id=current_doc,
            source_file=meta.get("source_file"),
            score=float(score),
            chunk_text=meta.get("chunk_text"),
            chunk_index=meta.get("chunk_index")
        ))
        
        if len(results) >= top_k:
            break
            
    return results
