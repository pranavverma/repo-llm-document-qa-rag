import json
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever


class DocumentStore:
    """
    Persistent FAISS vector store with per-document tracking.

    Each ingested file is assigned a unique doc_id. Chunks are saved to disk
    individually (under .vectorstore/chunks/) so the FAISS index can be rebuilt
    from scratch when a document is deleted — FAISS doesn't support in-place
    deletion by metadata.

    Layout on disk:
        .vectorstore/
            metadata.json       ← document registry
            chunks/
                <doc_id>.json   ← raw chunks for that document
            faiss_index/        ← FAISS binary files
    """

    _META_FILE = "metadata.json"
    _CHUNKS_DIR = "chunks"
    _INDEX_DIR = "faiss_index"

    def __init__(self, store_path: str, embeddings: Embeddings):
        self.root = Path(store_path)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / self._CHUNKS_DIR).mkdir(exist_ok=True)

        self.embeddings = embeddings
        self._meta: Dict = self._load_meta()
        self._index: Optional[FAISS] = self._load_index()

    # ------------------------------------------------------------------ #
    #  Persistence helpers                                                  #
    # ------------------------------------------------------------------ #

    def _load_meta(self) -> Dict:
        path = self.root / self._META_FILE
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {"documents": {}}

    def _save_meta(self) -> None:
        with open(self.root / self._META_FILE, "w") as f:
            json.dump(self._meta, f, indent=2)

    def _load_index(self) -> Optional[FAISS]:
        index_dir = self.root / self._INDEX_DIR
        if index_dir.exists():
            return FAISS.load_local(
                str(index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        return None

    def _save_index(self) -> None:
        if self._index:
            self._index.save_local(str(self.root / self._INDEX_DIR))

    def _rebuild_index(self) -> None:
        """Reconstruct FAISS from all persisted chunk files after a deletion."""
        all_chunks: List[Document] = []
        for chunk_file in (self.root / self._CHUNKS_DIR).glob("*.json"):
            with open(chunk_file) as f:
                for item in json.load(f):
                    all_chunks.append(
                        Document(page_content=item["content"], metadata=item["metadata"])
                    )

        index_dir = self.root / self._INDEX_DIR
        if all_chunks:
            self._index = FAISS.from_documents(all_chunks, self.embeddings)
            self._save_index()
        else:
            self._index = None
            if index_dir.exists():
                shutil.rmtree(index_dir)

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def add_document(self, chunks: List[Document], filename: str) -> str:
        doc_id = str(uuid.uuid4())

        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["filename"] = filename

        # Persist raw chunks for future index rebuilds
        chunk_path = self.root / self._CHUNKS_DIR / f"{doc_id}.json"
        with open(chunk_path, "w") as f:
            json.dump(
                [{"content": c.page_content, "metadata": c.metadata} for c in chunks], f
            )

        if self._index is None:
            self._index = FAISS.from_documents(chunks, self.embeddings)
        else:
            self._index.add_documents(chunks)

        self._meta["documents"][doc_id] = {
            "filename": filename,
            "chunk_count": len(chunks),
        }
        self._save_index()
        self._save_meta()
        return doc_id

    def remove_document(self, doc_id: str) -> bool:
        if doc_id not in self._meta["documents"]:
            return False

        chunk_path = self.root / self._CHUNKS_DIR / f"{doc_id}.json"
        if chunk_path.exists():
            chunk_path.unlink()

        del self._meta["documents"][doc_id]
        self._save_meta()
        self._rebuild_index()
        return True

    def list_documents(self) -> List[Dict]:
        return [{"doc_id": k, **v} for k, v in self._meta["documents"].items()]

    def is_empty(self) -> bool:
        return self._index is None or not self._meta["documents"]

    def get_retriever(self, top_k: int = 4) -> Optional[VectorStoreRetriever]:
        if self._index is None:
            return None
        return self._index.as_retriever(search_kwargs={"k": top_k})
