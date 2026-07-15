import os
import hashlib
import pickle
import time
from typing import List
import numpy as np
import google.generativeai as genai

_configured = False
_cache_file = "data/embedding_cache.pkl"
_cache = {}
_local_model = None

def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_model

def local_embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    model = _get_local_model()
    return model.encode(texts, convert_to_numpy=True).astype(np.float32)

def local_embed_query(query: str) -> np.ndarray:
    model = _get_local_model()
    return model.encode([query], convert_to_numpy=True).astype(np.float32)

def _load_cache():
    global _cache
    from storage.remote_sync import pull_from_supabase
    
    try:
        pull_from_supabase(_cache_file, "embedding_cache.pkl")
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Error pulling cache from Supabase: {e}")
        
    if os.path.exists(_cache_file):
        try:
            with open(_cache_file, "rb") as f:
                _cache = pickle.load(f)
        except Exception:
            _cache = {}

def _save_cache():
    os.makedirs(os.path.dirname(_cache_file), exist_ok=True)
    temp_file = _cache_file + ".tmp"
    with open(temp_file, "wb") as f:
        pickle.dump(_cache, f)
    os.replace(temp_file, _cache_file)
    
    from storage.remote_sync import push_to_supabase
    try:
        push_to_supabase(_cache_file, "embedding_cache.pkl")
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Error pushing cache to Supabase: {e}")

def _get_cache_key(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

_load_cache()

def configure_gemini():
    global _configured
    if not _configured:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing")
        genai.configure(api_key=api_key)
        _configured = True

import time

class EmbeddingRateLimitError(Exception): pass

def _call_gemini_embed_with_backoff(model: str, content: List[str], task_type: str):
    delays = [1, 2, 4, 8]
    for attempt, delay in enumerate(delays + [None]):
        try:
            return genai.embed_content(
                model=model,
                content=content,
                task_type=task_type
            )
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "rate limit" in error_str or "exhausted" in error_str:
                if delay is not None:
                    time.sleep(delay)
                    continue
                else:
                    raise EmbeddingRateLimitError(f"Gemini API rate limit exceeded after {len(delays)} retries: {str(e)}")
            raise

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a batch of texts using Gemini API, with local caching and backoff.
    """
    if not texts:
        return np.empty((0, 3072), dtype=np.float32)
        
    uncached_texts = []
    uncached_indices = []
    
    for i, text in enumerate(texts):
        key = _get_cache_key(text)
        if key not in _cache:
            uncached_texts.append(text)
            uncached_indices.append(i)
            
    if uncached_texts:
        configure_gemini()
        result = _call_gemini_embed_with_backoff(
            model="models/gemini-embedding-001",
            content=uncached_texts,
            task_type="retrieval_document"
        )
        for i, text in enumerate(uncached_texts):
            key = _get_cache_key(text)
            _cache[key] = np.array(result['embedding'][i], dtype=np.float32)
        _save_cache()
        
    embeddings = []
    for text in texts:
        key = _get_cache_key(text)
        embeddings.append(_cache[key])
        
    return np.array(embeddings, dtype=np.float32)

def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string using Gemini API, with local caching and backoff.
    """
    key = _get_cache_key(query)
    if key not in _cache:
        configure_gemini()
        result = _call_gemini_embed_with_backoff(
            model="models/gemini-embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        _cache[key] = np.array(result['embedding'], dtype=np.float32)
        _save_cache()
        
    return np.array([_cache[key]], dtype=np.float32)
