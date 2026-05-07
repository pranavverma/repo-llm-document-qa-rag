from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    message: str


class Source(BaseModel):
    filename: str
    page: Optional[int] = None
    excerpt: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=4, ge=1, le=20)


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    model: str
    documents_indexed: int
