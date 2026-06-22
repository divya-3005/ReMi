from typing import List
import numpy as np

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # Cache the model instance globally
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a batch of texts using sentence-transformers.
    Handles variable length text properly (truncates to max sequence length automatically).
    
    Args:
        texts: List of strings.
        
    Returns:
        np.ndarray of shape (len(texts), embedding_dim) as float32.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
        
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.astype(np.float32)

def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.
    
    Args:
        query: Query string.
        
    Returns:
        np.ndarray of shape (1, embedding_dim) as float32.
    """
    model = get_model()
    embedding = model.encode([query], convert_to_numpy=True)
    return embedding.astype(np.float32)
