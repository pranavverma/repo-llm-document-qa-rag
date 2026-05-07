from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

import api.state as state
from api.schemas import QueryRequest, QueryResponse, Source
from src.generation.chain import build_rag_chain, get_answer, stream_answer

router = APIRouter(prefix="/query", tags=["query"])


def _get_chain(top_k: int):
    if state.doc_store.is_empty():
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a file first.")
    retriever = state.doc_store.get_retriever(top_k=top_k)
    return build_rag_chain(
        retriever=retriever,
        model=state.cfg["ollama"]["model"],
        base_url=state.cfg["ollama"]["base_url"],
        temperature=state.cfg["ollama"]["temperature"],
    ), retriever


@router.post("/", response_model=QueryResponse)
def query(req: QueryRequest):
    chain, retriever = _get_chain(req.top_k)
    answer = get_answer(chain, req.question)

    # fetch source docs separately for citation metadata
    source_docs = retriever.invoke(req.question)
    sources = [
        Source(
            filename=d.metadata.get("filename", "unknown"),
            page=d.metadata.get("page"),
            excerpt=d.page_content[:250].strip(),
        )
        for d in source_docs
    ]

    return QueryResponse(question=req.question, answer=answer, sources=sources)


@router.post("/stream")
def query_stream(req: QueryRequest):
    """
    Server-sent token stream. Each chunk in the response is a raw text token
    from the LLM. Clients can read line-by-line or accumulate the full string.
    """
    chain, _ = _get_chain(req.top_k)

    def token_gen():
        for token in stream_answer(chain, req.question):
            yield token

    return StreamingResponse(token_gen(), media_type="text/plain")
