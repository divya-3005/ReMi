import os
from typing import List
import numpy as np
import google.generativeai as genai

_configured = False

def configure_gemini():
    global _configured
    if not _configured:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing")
        genai.configure(api_key=api_key)
        _configured = True

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a batch of texts using Gemini API.
    """
    if not texts:
        return np.empty((0, 768), dtype=np.float32)
        
    configure_gemini()
    result = genai.embed_content(
        model="models/embedding-001",
        content=texts,
        task_type="retrieval_document"
    )
    return np.array(result['embedding'], dtype=np.float32)

def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string using Gemini API.
    """
    configure_gemini()
    result = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    return np.array([result['embedding']], dtype=np.float32)
