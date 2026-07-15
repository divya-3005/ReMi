from typing import List
import numpy as np

_model = None

def get_model():
    global _model
    if _model is None:
        # Use fastembed instead of sentence-transformers (PyTorch)
        # fastembed uses ONNX under the hood and requires < 150MB of RAM
        # BAAI/bge-small-en-v1.5 produces exactly 384 dimensions matching our FAISS setup.
        from fastembed import TextEmbedding
        _model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _model

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a batch of texts using fastembed.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
        
    model = get_model()
    # fastembed returns a generator of embeddings, convert to stacked numpy array
    embeddings_list = list(model.embed(texts))
    return np.vstack(embeddings_list).astype(np.float32)

def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.
    """
    model = get_model()
    embeddings_list = list(model.embed([query]))
    return np.vstack(embeddings_list).astype(np.float32)
