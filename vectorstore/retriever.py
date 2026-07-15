from dataclasses import dataclass
from typing import List, Optional
from vectorstore.store import FaissStore
from vectorstore.embedder import embed_query, local_embed_query, EmbeddingRateLimitError

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
    Searches the vector store for the most relevant chunks using Hybrid Search (FAISS + BM25) and Reciprocal Rank Fusion.
    Gracefully handles dual FAISS indexes (Gemini + Local).
    """
    if store.total_chunks() == 0:
        import warnings
        warnings.warn("Vector store is empty.")
        return []
        
    search_k = min(store.total_chunks(), max(top_k * 10, 50))
    faiss_ranks = {}
    
    # 1a. FAISS Search (Primary - Gemini)
    if store.index_primary.ntotal > 0:
        try:
            query_emb_primary = embed_query(query)
            faiss_scores, faiss_indices = store.index_primary.search(query_emb_primary, search_k)
            for rank, faiss_idx in enumerate(faiss_indices[0]):
                if faiss_idx != -1:
                    global_idx = store.primary_meta_idx[faiss_idx]
                    faiss_ranks[global_idx] = rank + 1
        except EmbeddingRateLimitError:
            # If rate limited, just skip the primary index for this query
            pass

    # 1b. FAISS Search (Fallback - Local)
    if store.index_fallback.ntotal > 0:
        query_emb_fallback = local_embed_query(query)
        faiss_scores, faiss_indices = store.index_fallback.search(query_emb_fallback, search_k)
        for rank, faiss_idx in enumerate(faiss_indices[0]):
            if faiss_idx != -1:
                global_idx = store.fallback_meta_idx[faiss_idx]
                faiss_ranks[global_idx] = rank + 1

    # 2. BM25 Search (Sparse)
    bm25_ranks = {}
    if store.bm25:
        import numpy as np
        tokenized_query = query.lower().split()
        bm25_scores = store.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:search_k]
        for rank, idx in enumerate(top_bm25_indices):
            if bm25_scores[idx] > 0:
                bm25_ranks[idx] = rank + 1

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    all_indices = set(faiss_ranks.keys()).union(set(bm25_ranks.keys()))
    k_rrf = 60
    
    for idx in all_indices:
        score = 0.0
        if idx in faiss_ranks:
            score += 1.0 / (k_rrf + faiss_ranks[idx])
        if idx in bm25_ranks:
            score += 1.0 / (k_rrf + bm25_ranks[idx])
        rrf_scores[idx] = score
        
    # Sort indices by their RRF score descending
    sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
    
    results = []
    seen_docs = set()
    
    for idx in sorted_indices:
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
            score=float(rrf_scores[idx]),
            chunk_text=meta.get("chunk_text"),
            chunk_index=meta.get("chunk_index")
        ))
        
        if len(results) >= top_k:
            break
            
    return results
