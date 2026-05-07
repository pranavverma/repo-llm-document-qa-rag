import logging
from contextlib import asynccontextmanager

import httpx
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import api.state as state
from api.routes import documents, query
from api.schemas import HealthResponse
from src.retrieval.embedder import get_embeddings
from src.retrieval.store import DocumentStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.cfg = _load_config()
    embeddings = get_embeddings(state.cfg["embeddings"]["model"])
    state.doc_store = DocumentStore(state.cfg["storage"]["vector_store_path"], embeddings)
    logger.info(f"Store ready. Documents indexed: {len(state.doc_store.list_documents())}")
    yield


app = FastAPI(
    title="Document Q&A API",
    description=(
        "Upload PDFs and text files, then ask questions. "
        "Answers are grounded in your documents and include source citations. "
        "Powered by a local Ollama model — no data leaves your machine."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(query.router)


@app.get("/health", response_model=HealthResponse)
async def health():
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{state.cfg['ollama']['base_url']}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        ollama_reachable=ollama_ok,
        model=state.cfg.get("ollama", {}).get("model", "unknown"),
        documents_indexed=len(state.doc_store.list_documents()) if state.doc_store else 0,
    )
