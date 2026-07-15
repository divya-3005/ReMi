import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from vectorstore.embedder import embed_texts, local_embed_texts, EmbeddingRateLimitError
from rank_bm25 import BM25Okapi
from storage.remote_sync import pull_from_supabase

class FaissStore:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.index_primary_path = os.path.join(data_dir, "faiss_primary.index")
        self.index_fallback_path = os.path.join(data_dir, "faiss_fallback.index")
        self.meta_path = os.path.join(data_dir, "faiss_meta.json")
        self.bm25_path = os.path.join(data_dir, "bm25_corpus.json")
        
        self.dim_primary = 3072
        self.dim_fallback = 384
        
        # In-memory metadata list global across all chunks
        self.metadata: List[Dict[str, Any]] = []
        # In-memory BM25 corpus (tokenized texts)
        self.bm25_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        
        # Mapping from index-specific ID to global metadata ID
        self.primary_meta_idx: List[int] = []
        self.fallback_meta_idx: List[int] = []
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            
        # Pull persistent files from Supabase before attempting local load
        def _safe_pull(path, key):
            try:
                pull_from_supabase(path, key)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Error pulling {key} from Supabase: {e}")
                
        _safe_pull(self.index_primary_path, "faiss_primary.index")
        _safe_pull(self.index_fallback_path, "faiss_fallback.index")
        _safe_pull(self.meta_path, "faiss_meta.json")
        _safe_pull(self.bm25_path, "bm25_corpus.json")
            
        self.load()

    def _rebuild_meta_mappings(self):
        self.primary_meta_idx = []
        self.fallback_meta_idx = []
        for global_idx, meta in enumerate(self.metadata):
            if meta.get("embedding_source") == "fallback":
                self.fallback_meta_idx.append(global_idx)
            else:
                self.primary_meta_idx.append(global_idx)

    def load(self):
        """Loads FAISS indexes, metadata, and BM25 corpus."""
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

        if os.path.exists(self.index_primary_path):
            self.index_primary = faiss.read_index(self.index_primary_path)
        else:
            self.index_primary = faiss.IndexFlatIP(self.dim_primary)
            
        if os.path.exists(self.index_fallback_path):
            self.index_fallback = faiss.read_index(self.index_fallback_path)
        else:
            self.index_fallback = faiss.IndexFlatIP(self.dim_fallback)
            
        self._rebuild_meta_mappings()

        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "r", encoding="utf-8") as f:
                self.bm25_corpus = json.load(f)
            if self.bm25_corpus:
                self.bm25 = BM25Okapi(self.bm25_corpus)
        else:
            self.bm25_corpus = []
            self.bm25 = None

    def save(self):
        """Persists the FAISS indexes, metadata, and BM25 corpus to disk and pushes to Supabase."""
        from storage.remote_sync import push_to_supabase
        
        faiss.write_index(self.index_primary, self.index_primary_path)
        faiss.write_index(self.index_fallback, self.index_fallback_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        with open(self.bm25_path, "w", encoding="utf-8") as f:
            json.dump(self.bm25_corpus, f)
            
        # Push to Supabase after successful local save
        def _safe_push(path, key):
            try:
                push_to_supabase(path, key)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Error pushing {key} to Supabase: {e}")
                
        _safe_push(self.index_primary_path, "faiss_primary.index")
        _safe_push(self.index_fallback_path, "faiss_fallback.index")
        _safe_push(self.meta_path, "faiss_meta.json")
        _safe_push(self.bm25_path, "bm25_corpus.json")

    def total_chunks(self) -> int:
        """Returns the total number of chunks across both indexes."""
        return self.index_primary.ntotal + self.index_fallback.ntotal

    def add_document(self, doc_id: str, chunks: List[Any], source_file: str = "unknown"):
        """
        Embeds and adds chunks of a document to the correct FAISS index and BM25 corpus.
        Falls back to local embeddings if Gemini rate limit is hit.
        """
        if not chunks:
            return
            
        texts = [chunk.content for chunk in chunks]
        
        try:
            embeddings = embed_texts(texts)
            source = "primary"
        except EmbeddingRateLimitError:
            embeddings = local_embed_texts(texts)
            source = "fallback"
        
        faiss.normalize_L2(embeddings)
        
        target_index = self.index_primary if source == "primary" else self.index_fallback
        start_idx = target_index.ntotal
        
        target_index.add(embeddings)
        
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25_corpus.extend(tokenized_texts)
        if self.bm25_corpus:
            self.bm25 = BM25Okapi(self.bm25_corpus)
        
        for idx, chunk in enumerate(chunks):
            self.metadata.append({
                "faiss_index": start_idx + idx,
                "embedding_source": source,
                "chunk_id": chunk.chunk_id,
                "doc_id": doc_id,
                "source_file": source_file,
                "chunk_text": chunk.content,
                "token_count": chunk.token_count,
                "chunk_index": getattr(chunk, 'chunk_index', 0)
            })
            
        self._rebuild_meta_mappings()
        self.save()

    def remove_document(self, doc_id: str):
        """
        Removes all chunks associated with doc_id from the store.
        """
        if len(self.metadata) == 0:
            return
            
        new_metadata = []
        new_primary_embeddings = []
        new_fallback_embeddings = []
        new_bm25_corpus = []
        
        for global_idx, meta in enumerate(self.metadata):
            if meta.get("doc_id") != doc_id:
                source = meta.get("embedding_source", "primary")
                pos = meta.get("faiss_index")
                
                if source == "fallback":
                    vec = self.index_fallback.reconstruct(pos)
                    meta["faiss_index"] = len(new_fallback_embeddings)
                    new_fallback_embeddings.append(vec)
                else:
                    vec = self.index_primary.reconstruct(pos)
                    meta["faiss_index"] = len(new_primary_embeddings)
                    new_primary_embeddings.append(vec)
                    
                new_metadata.append(meta)
                new_bm25_corpus.append(self.bm25_corpus[global_idx])
                
        self.index_primary = faiss.IndexFlatIP(self.dim_primary)
        self.index_fallback = faiss.IndexFlatIP(self.dim_fallback)
        self.metadata = new_metadata
        self.bm25_corpus = new_bm25_corpus
        
        if self.bm25_corpus:
            self.bm25 = BM25Okapi(self.bm25_corpus)
        else:
            self.bm25 = None
        
        if new_primary_embeddings:
            self.index_primary.add(np.vstack(new_primary_embeddings).astype(np.float32))
        if new_fallback_embeddings:
            self.index_fallback.add(np.vstack(new_fallback_embeddings).astype(np.float32))
            
        self._rebuild_meta_mappings()
        self.save()
