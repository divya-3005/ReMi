import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from vectorstore.embedder import embed_texts
from rank_bm25 import BM25Okapi

class FaissStore:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.index_path = os.path.join(data_dir, "faiss.index")
        self.meta_path = os.path.join(data_dir, "faiss_meta.json")
        self.bm25_path = os.path.join(data_dir, "bm25_corpus.json")
        self.dim = 3072
        
        # In-memory metadata list parallel to FAISS index
        self.metadata: List[Dict[str, Any]] = []
        # In-memory BM25 corpus (tokenized texts)
        self.bm25_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            
        self.load()

    def load(self):
        """Loads FAISS index, metadata, and BM25 corpus."""
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadata = []

        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "r", encoding="utf-8") as f:
                self.bm25_corpus = json.load(f)
            if self.bm25_corpus:
                self.bm25 = BM25Okapi(self.bm25_corpus)
        else:
            self.bm25_corpus = []
            self.bm25 = None

    def save(self):
        """Persists the FAISS index, metadata, and BM25 corpus to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        with open(self.bm25_path, "w", encoding="utf-8") as f:
            json.dump(self.bm25_corpus, f)

    def total_chunks(self) -> int:
        """Returns the total number of vectors in the store."""
        return self.index.ntotal

    def add_document(self, doc_id: str, chunks: List[Any], source_file: str = "unknown"):
        """
        Embeds and adds chunks of a document to the FAISS index and BM25 corpus.
        """
        if not chunks:
            return
            
        texts = [chunk.content for chunk in chunks]
        embeddings = embed_texts(texts)
        
        # Normalize vectors for cosine similarity via IndexFlatIP
        faiss.normalize_L2(embeddings)
        
        start_idx = self.index.ntotal
        
        # Add to FAISS
        self.index.add(embeddings)
        
        # Tokenize and add to BM25 Corpus
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25_corpus.extend(tokenized_texts)
        if self.bm25_corpus:
            self.bm25 = BM25Okapi(self.bm25_corpus)
        
        # Add metadata
        for idx, chunk in enumerate(chunks):
            self.metadata.append({
                "faiss_index": start_idx + idx,
                "chunk_id": chunk.chunk_id,
                "doc_id": doc_id,
                "source_file": source_file,
                "chunk_text": chunk.content[:500],  # Increased to 500 chars for context
                "token_count": chunk.token_count,
                "chunk_index": getattr(chunk, 'chunk_index', 0)
            })
            
        self.save()

    def remove_document(self, doc_id: str):
        """
        Removes all chunks associated with doc_id from the store.
        """
        if self.index.ntotal == 0:
            return
            
        new_metadata = []
        new_embeddings_list = []
        new_bm25_corpus = []
        
        for idx, meta in enumerate(self.metadata):
            if meta.get("doc_id") != doc_id:
                # Reconstruct vector using the explicitly tracked position
                pos = meta.get("faiss_index")
                if pos is None:
                    pos = idx
                vec = self.index.reconstruct(pos)
                
                # Update the position to be contiguous for the rebuilt index
                meta["faiss_index"] = len(new_metadata)
                new_metadata.append(meta)
                new_embeddings_list.append(vec)
                new_bm25_corpus.append(self.bm25_corpus[pos])
                
        # Rebuild index
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = new_metadata
        self.bm25_corpus = new_bm25_corpus
        
        if self.bm25_corpus:
            self.bm25 = BM25Okapi(self.bm25_corpus)
        else:
            self.bm25 = None
        
        if new_embeddings_list:
            new_embeddings_array = np.vstack(new_embeddings_list).astype(np.float32)
            self.index.add(new_embeddings_array)
            
        self.save()
