import os
import json
import dataclasses
from typing import List, Dict, Any, Optional, Tuple
from models.document import Document, Chunk

class DocumentStore:
    """A simple JSON-based local document store."""

    def __init__(self, store_path: str = "data/store.json"):
        """Initializes the document store and creates parent directories if needed.

        Args:
            store_path: Path to the local JSON file where documents are stored.
        """
        self.store_path = store_path
        db_dir = os.path.dirname(self.store_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._initialize_store()

    def _initialize_store(self) -> None:
        """Creates the store file with empty dict structures if it does not exist."""
        if not os.path.exists(self.store_path):
            self._write_store({"documents": {}, "chunks": {}})

    def _read_store(self) -> Dict[str, Any]:
        """Reads the local JSON store file. Returns empty structure on failure."""
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"documents": {}, "chunks": {}}

    def _write_store(self, data: Dict[str, Any]) -> None:
        """Writes the dictionary back to the local JSON store."""
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save(self, doc: Document, chunks: List[Chunk]) -> None:
        """Saves a document and its corresponding chunks to the store.

        Args:
            doc: Document object containing raw text and metadata.
            chunks: List of Chunk objects associated with the document.
        """
        data = self._read_store()
        data["documents"][doc.id] = dataclasses.asdict(doc)
        for chunk in chunks:
            data["chunks"][chunk.chunk_id] = dataclasses.asdict(chunk)
        self._write_store(data)

    def get_by_id(self, doc_id: str) -> Optional[Tuple[Document, List[Chunk]]]:
        """Retrieves a document and its chunks by document ID.

        Args:
            doc_id: The ID of the document to retrieve.

        Returns:
            A tuple of (Document, list of Chunks) if found, otherwise None.
        """
        data = self._read_store()
        doc_dict = data["documents"].get(doc_id)
        if not doc_dict:
            return None

        doc = Document(**doc_dict)
        chunks = []
        for chunk_dict in data["chunks"].values():
            if chunk_dict["document_id"] == doc_id:
                chunks.append(Chunk(**chunk_dict))

        chunks.sort(key=lambda x: x.char_offset)
        return doc, chunks

    def list_all(self) -> List[Document]:
        """Lists all ingested documents in the store.

        Returns:
            A list of Document objects.
        """
        data = self._read_store()
        docs = []
        for doc_dict in data["documents"].values():
            docs.append(Document(**doc_dict))
        return docs

    def get_chunk_count(self, doc_id: str) -> int:
        """Retrieves the count of chunks for a given document ID.

        Args:
            doc_id: The document ID.

        Returns:
            Integer count of chunks.
        """
        data = self._read_store()
        count = 0
        for chunk_dict in data["chunks"].values():
            if chunk_dict["document_id"] == doc_id:
                count += 1
        return count

    def delete(self, doc_id: str) -> bool:
        """Deletes a document and its associated chunks from the store.

        Args:
            doc_id: The ID of the document to delete.

        Returns:
            True if deleted successfully, False if document was not found.
        """
        data = self._read_store()
        if doc_id not in data["documents"]:
            return False

        del data["documents"][doc_id]
        data["chunks"] = {
            cid: c for cid, c in data["chunks"].items() if c["document_id"] != doc_id
        }

        self._write_store(data)
        return True
