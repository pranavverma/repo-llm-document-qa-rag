import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

import api.state as state
from api.schemas import DocumentInfo, UploadResponse
from src.ingestion.chunker import split_documents
from src.ingestion.loader import SUPPORTED, load_file

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED)}",
        )

    contents = await file.read()

    # write to a temp file so the file-based loaders (PyPDFLoader etc.) can open it
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        docs = load_file(tmp_path)
        chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk in chunks:
            chunk.metadata["filename"] = file.filename
        doc_id = state.doc_store.add_document(chunks, file.filename)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        chunk_count=len(chunks),
        message="Document indexed successfully.",
    )


@router.get("/", response_model=list[DocumentInfo])
def list_documents():
    return state.doc_store.list_documents()


@router.delete("/{doc_id}")
def delete_document(doc_id: str):
    if not state.doc_store.remove_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"message": "Document removed and index rebuilt."}
