from langchain_core.documents import Document
from src.ingestion.chunker import split_documents


def test_chunks_respect_size_limit():
    doc = Document(page_content="word " * 400, metadata={"source": "test.txt"})
    chunks = split_documents([doc], chunk_size=100, chunk_overlap=10)
    for chunk in chunks:
        # slight tolerance: splitter may slightly exceed chunk_size at word boundaries
        assert len(chunk.page_content) <= 130


def test_chunk_count_scales_with_content():
    doc = Document(page_content="sentence text. " * 200, metadata={})
    chunks = split_documents([doc], chunk_size=150, chunk_overlap=20)
    assert len(chunks) >= 5


def test_metadata_is_preserved():
    doc = Document(
        page_content="hello world " * 60,
        metadata={"source": "report.pdf", "page": 3},
    )
    chunks = split_documents([doc], chunk_size=100, chunk_overlap=10)
    for chunk in chunks:
        assert chunk.metadata["source"] == "report.pdf"
        assert chunk.metadata["page"] == 3


def test_empty_document_returns_no_chunks():
    doc = Document(page_content="   ", metadata={})
    chunks = split_documents([doc], chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 0
