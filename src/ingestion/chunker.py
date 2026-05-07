from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Split documents into overlapping chunks for retrieval.

    RecursiveCharacterTextSplitter tries paragraph breaks first, then
    sentence boundaries, then word boundaries — so chunks rarely cut
    mid-sentence, which keeps retrieved context coherent.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)
