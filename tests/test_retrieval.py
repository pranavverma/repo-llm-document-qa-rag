from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


@pytest.fixture
def mock_store(tmp_path):
    embeddings = MagicMock()
    embeddings.embed_documents.return_value = [[0.1] * 384]
    embeddings.embed_query.return_value = [0.1] * 384

    with patch("src.retrieval.store.FAISS") as mock_faiss:
        mock_index = MagicMock()
        mock_faiss.from_documents.return_value = mock_index
        mock_index.save_local = MagicMock()

        from src.retrieval.store import DocumentStore
        store = DocumentStore(str(tmp_path), embeddings)
        store._index = None  # start empty
        yield store, mock_faiss


def test_store_is_empty_on_init(mock_store):
    store, _ = mock_store
    assert store.is_empty()


def test_add_document_registers_metadata(mock_store):
    store, _ = mock_store
    chunks = [Document(page_content="hello world", metadata={})]
    doc_id = store.add_document(chunks, "sample.txt")

    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0]["filename"] == "sample.txt"
    assert docs[0]["doc_id"] == doc_id
    assert docs[0]["chunk_count"] == 1


def test_add_document_injects_doc_id_into_chunks(mock_store):
    store, _ = mock_store
    chunks = [Document(page_content="content", metadata={})]
    doc_id = store.add_document(chunks, "file.txt")
    assert chunks[0].metadata["doc_id"] == doc_id
    assert chunks[0].metadata["filename"] == "file.txt"


def test_chunks_are_persisted_to_disk(mock_store, tmp_path):
    store, _ = mock_store
    chunks = [Document(page_content="persistent content", metadata={})]
    doc_id = store.add_document(chunks, "persist.txt")

    chunk_file = tmp_path / "chunks" / f"{doc_id}.json"
    assert chunk_file.exists()
