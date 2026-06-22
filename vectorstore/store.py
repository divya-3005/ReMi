import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any
from vectorstore.embedder import embed_texts

class FaissStore:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.index_path = os.path.join(data_dir, "faiss.index")
        self.meta_path = os.path.join(data_dir, "faiss_meta.json")
        self.dim = 384
        
        # In-memory metadata list parallel to FAISS index
        self.metadata: List[Dict[str, Any]] = []
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            
        self.load()

    def load(self):
        """Loads FAISS index and metadata. Initializes empty if not found."""
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadata = []

    def save(self):
        """Persists the FAISS index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def total_chunks(self) -> int:
        """Returns the total number of vectors in the store."""
        return self.index.ntotal

    def add_document(self, doc_id: str, chunks: List[Any], source_file: str = "unknown"):
        """
        Embeds and adds chunks of a document to the FAISS index.
        """
        if not chunks:
            return
            
        texts = [chunk.content for chunk in chunks]
        embeddings = embed_texts(texts)
        
        # Normalize vectors for cosine similarity via IndexFlatIP
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS
        self.index.add(embeddings)
        
        # Add metadata
        for chunk in chunks:
            self.metadata.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": doc_id,
                "source_file": source_file,
                "chunk_text": chunk.content[:200],  # store up to 200 chars
                "token_count": chunk.token_count,
                "chunk_index": getattr(chunk, 'chunk_index', 0)
            })
            
        self.save()

    def remove_document(self, doc_id: str):
        """
        Removes all chunks associated with doc_id from the store.
        Rebuilds the index from the remaining metadata and embeddings.
        """
        if self.index.ntotal == 0:
            return
            
        new_metadata = []
        new_embeddings_list = []
        
        for i, meta in enumerate(self.metadata):
            if meta.get("doc_id") != doc_id:
                new_metadata.append(meta)
                # Reconstruct vector i
                vec = self.index.reconstruct(i)
                new_embeddings_list.append(vec)
                
        # Rebuild index
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = new_metadata
        
        if new_embeddings_list:
            new_embeddings_array = np.vstack(new_embeddings_list).astype(np.float32)
            self.index.add(new_embeddings_array)
            
        self.save()
