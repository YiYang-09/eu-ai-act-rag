"""Cross-encoder reranking with an on-disk score cache."""

from __future__ import annotations

import json
from pathlib import Path

from .dense import INDEX_DIR, chunk_text

RERANKERS = {"bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3"}


class Reranker:
    def __init__(self, key: str = "bge-reranker-v2-m3", device: str | None = None, max_length: int = 512,
                 cache_path: Path | None = None):
        self.key = key
        self.device = device
        self.max_length = max_length
        self._model = None
        self.cache_path = cache_path or INDEX_DIR / f"rerank_cache_{key}.json"
        self.cache: dict[str, float] = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(RERANKERS[self.key], device=self.device, max_length=self.max_length)
        return self._model

    def rerank(self, query: str, chunks: list[dict], k: int = 10, query_id: str | None = None) -> list[tuple[dict, float]]:
        keys = [f"{query_id or query}||{c['strategy']}||{c['chunk_id']}" for c in chunks]
        missing = [i for i, key in enumerate(keys) if key not in self.cache]
        if missing:
            scores = self._load().predict([(query, chunk_text(chunks[i])) for i in missing], batch_size=8)
            for i, s in zip(missing, scores):
                self.cache[keys[i]] = float(s)
        ranked = sorted(zip(chunks, (self.cache[key] for key in keys)), key=lambda x: -x[1])
        return ranked[:k]

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache))
