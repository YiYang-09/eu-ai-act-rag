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
        self._unsaved = 0
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
            self._unsaved += len(missing)
            if self._unsaved >= 150:  # checkpoint so an interrupted run can resume
                self.save()
        ranked = sorted(zip(chunks, (self.cache[key] for key in keys)), key=lambda x: -x[1])
        return ranked[:k]

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache))
        self._unsaved = 0


def limit_recitals(ranked: list[dict], k: int, max_recitals: int | None) -> list[dict]:
    """Take the top k, allowing at most max_recitals recital chunks.

    Recitals explain the law but do not bind; a compliance answer must rest on articles
    and annexes, yet recitals restate the topic in plain words and so win the reranker.
    """
    if max_recitals is None:
        return ranked[:k]
    out, n_rct = [], 0
    for c in ranked:
        if c["kind"] == "recital":
            if n_rct >= max_recitals:
                continue
            n_rct += 1
        out.append(c)
        if len(out) == k:
            break
    return out


def pin_chunks(top: list[dict], ranked: list[dict], pinned: list[str], k: int) -> list[dict]:
    """Reserve slots in the top k for pinned chunks (e.g. the application-date article that graph
    expansion adds for a "when does ... apply" question). Cross-encoder scores saturate near 1.0 for
    Swedish queries, so the reranker alone can push such a chunk to rank 20+. A pinned chunk that is
    not already in the top k replaces the lowest-ranked non-pinned entry."""
    out = list(top[:k])
    ids = {c["chunk_id"] for c in out}
    by_id = {c["chunk_id"]: c for c in ranked}
    for pid in pinned:
        if pid in ids or pid not in by_id:
            continue
        for i in range(len(out) - 1, -1, -1):
            if out[i]["chunk_id"] not in pinned:
                ids.discard(out[i]["chunk_id"])
                out[i] = by_id[pid]
                ids.add(pid)
                break
    return out
