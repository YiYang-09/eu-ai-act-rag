"""FastAPI app.

  GET  /health                   service + LLM status
  POST /search {question, lang?, role?}   retrieval only (no LLM)
  POST /ask    {question, lang?, role?}   answer with numbered, linked citations
  GET  /roles                    roles and the documents each may see

Run:  uvicorn lexrag.serve.api:app --host 0.0.0.0 --port 8000
Configuration comes from environment variables (see build_service_from_env).
"""

from __future__ import annotations

import os
import threading
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .service import QAService, detect_lang, load_access, load_chunks


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    lang: str | None = Field(default=None, pattern="^(en|sv)$")
    role: str = "public"


def build_service_from_env() -> QAService:
    """LEXRAG_STORE=qdrant|numpy, QDRANT_URL or QDRANT_PATH (local file mode, no server), OLLAMA_URL, LLM_MODEL, EMBED_DEVICE, LEXRAG_FAKE=1 (tests)."""
    if os.getenv("LEXRAG_FAKE") == "1":
        from .fakes import fake_service

        return fake_service()
    from ..retrieval.dense import Embedder
    from ..retrieval.rerank import Reranker
    from .llm import OllamaLLM
    from .store import NumpyStore, QdrantStore

    device = os.getenv("EMBED_DEVICE") or None
    embedder = Embedder("bge-m3", device)
    chunks = load_chunks()
    if os.getenv("LEXRAG_STORE", "qdrant") == "qdrant":
        if os.getenv("QDRANT_PATH"):  # embedded Qdrant, no server or Docker needed
            store = QdrantStore(path=os.getenv("QDRANT_PATH"))
        else:
            store = QdrantStore(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    else:  # in-process store from the cached evaluation vectors
        from ..retrieval.dense import build_or_load

        store = NumpyStore(1024)
        for lang in ("en", "sv"):
            cs = [c for c in chunks if c["lang"] == lang]
            store.upsert(cs, build_or_load(cs, "bge-m3", "structural", lang, embedder))
    llm = OllamaLLM(model=os.getenv("LLM_MODEL", "qwen2.5:7b"), url=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    return QAService(chunks, store, embedder, Reranker(device=device, cache_path=None), llm, load_access())


def create_app(service: QAService | None = None) -> FastAPI:
    app = FastAPI(title="EU AI Act & GDPR compliance assistant", version="0.1.0")
    state = {"svc": service}
    lock = threading.Lock()  # concurrent first requests must not build two services (embedded Qdrant is single-client)

    def svc() -> QAService:
        if state["svc"] is None:
            with lock:
                if state["svc"] is None:
                    state["svc"] = build_service_from_env()
        return state["svc"]

    @app.on_event("startup")
    def warm_up():
        if os.getenv("LEXRAG_EAGER", "1") == "1" and service is None:
            svc()  # load models before the first request instead of during it

    @app.get("/health")
    def health():
        s = svc()
        return {"status": "ok", "llm": getattr(s.llm, "model", "?"), "llm_reachable": s.llm.healthy(),
                "chunks": {k: len(v) for k, v in s.chunks.items()}, "graph": sorted(s.expanders)}

    @app.get("/roles")
    def roles():
        return svc().access

    @app.post("/search")
    def search(q: Query):
        s = svc()
        lang = q.lang or detect_lang(q.question)
        try:
            hits, timings = s.retrieve(q.question, lang, q.role)
        except PermissionError as e:
            raise HTTPException(403, str(e))
        return {"lang": lang, "timings_ms": {k: round(v * 1000) for k, v in timings.items()},
                "results": [{"chunk_id": c["chunk_id"], "provision_id": c["provision_ids"][0], "header": c["header"],
                             "text": c["text"], "score": round(sc, 4), "via": via} for c, sc, via in hits]}

    @app.post("/ask")
    def ask(q: Query):
        try:
            return asdict(svc().ask(q.question, q.lang, q.role))
        except PermissionError as e:
            raise HTTPException(403, str(e))

    return app


app = create_app()
