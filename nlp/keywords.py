from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_keywords(chunks_data: List[Dict[str, str]], top_n: int = 10) -> Dict[str, Any]:
    """
    Extracts keywords from a list of chunks using TF-IDF.
    
    Args:
        chunks_data: List of dicts with 'chunk_id' and 'content'.
        top_n: Number of keywords to return per chunk and globally.
        
    Returns:
        Dict with structure:
        {
            "per_chunk": { "chunk_id": [{"keyword": "...", "score": 0.5}, ...] },
            "document_level": [{"keyword": "...", "score": 0.5}, ...]
        }
    """
    if not chunks_data:
        return {"per_chunk": {}, "document_level": []}
        
    texts = [c["content"] for c in chunks_data]
    chunk_ids = [c["chunk_id"] for c in chunks_data]
    
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # Handle cases where vocabulary is empty
        return {"per_chunk": {cid: [] for cid in chunk_ids}, "document_level": []}
        
    feature_names = vectorizer.get_feature_names_out()
    
    per_chunk = {}
    doc_scores = np.zeros(len(feature_names))
    
    for row, cid in enumerate(chunk_ids):
        row_data = tfidf_matrix.getrow(row)
        dense_row = row_data.toarray().flatten()
        doc_scores += dense_row
        
        top_indices = dense_row.argsort()[-top_n:][::-1]
        
        chunk_keywords = []
        for idx in top_indices:
            score = dense_row[idx]
            if score > 0:
                chunk_keywords.append({
                    "keyword": feature_names[idx],
                    "score": float(score)
                })
        per_chunk[cid] = chunk_keywords
        
    # Average across all chunks
    doc_scores /= len(texts)
    top_doc_indices = doc_scores.argsort()[-top_n:][::-1]
    
    document_level = []
    for idx in top_doc_indices:
        score = doc_scores[idx]
        if score > 0:
            document_level.append({
                "keyword": feature_names[idx],
                "score": float(score)
            })
            
    return {
        "per_chunk": per_chunk,
        "document_level": document_level
    }
