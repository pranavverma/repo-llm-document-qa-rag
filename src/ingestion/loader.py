from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED = {".pdf", ".txt", ".md"}


def load_file(path: str) -> List[Document]:
    """
    Load a document from disk into LangChain Document objects.

    PDFs are loaded page-by-page so each chunk can carry a page number in
    its metadata — useful for citations. Text/markdown files are loaded as
    a single document.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        return PyPDFLoader(str(p)).load()
    elif suffix in {".txt", ".md"}:
        return TextLoader(str(p), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {SUPPORTED}")
