from typing import Optional
from src.retrieval.store import DocumentStore

# Populated at startup in api/main.py lifespan handler.
# Routes import from here to avoid circular imports.
doc_store: Optional[DocumentStore] = None
cfg: dict = {}
