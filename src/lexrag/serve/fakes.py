"""Small deterministic stand-ins so the whole service runs in tests and CI without models."""

from __future__ import annotations

import hashlib

import numpy as np

from ..retrieval.bm25 import tokenize
from .llm import FakeLLM
from .service import QAService, load_access, load_chunks
from .store import NumpyStore

DIM = 256


class HashEmbedder:
    """Bag-of-words hashed into 256 dims, L2-normalised. Enough to make dense search non-random."""

    def _vec(self, text: str, lang: str = "en") -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for tok in tokenize(text, lang):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_queries(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def encode_chunks(self, chunks):
        return np.stack([self._vec(f"{c['header']} {c['text']}", c["lang"]) for c in chunks])


class OverlapReranker:
    """Scores by token overlap with the question."""

    def rerank(self, query, chunks, k=10, query_id=None):
        q = set(tokenize(query, "en")) | set(tokenize(query, "sv"))
        scored = [(c, len(q & set(tokenize(c["header"] + " " + c["text"], c["lang"]))) / (1 + len(q)))
                  for c in chunks]
        return sorted(scored, key=lambda x: -x[1])[:k]


def fake_service(chunks: list[dict] | None = None, use_graph: bool = True) -> QAService:
    chunks = chunks if chunks is not None else load_chunks()
    emb = HashEmbedder()
    store = NumpyStore(DIM)
    store.upsert(chunks, emb.encode_chunks(chunks))
    return QAService(chunks, store, emb, OverlapReranker(), FakeLLM(), load_access(), use_graph=use_graph)
