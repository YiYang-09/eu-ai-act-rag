"""Vector stores behind one interface: Qdrant for deployment, NumPy for tests and offline runs.

Both support the two things the service needs:
  search(qvec, lang, k, docs)   dense search, filtered by language and allowed documents
  vectors(chunk_ids)            fetch stored vectors (used by graph expansion)
and the incremental-ingestion primitives:
  hashes(lang)                  {chunk_id: content hash} currently stored
  upsert(chunks, vectors) / delete(chunk_ids)
"""

from __future__ import annotations

import hashlib
import uuid

import numpy as np

COLLECTION = "eu_law_chunks"
PAYLOAD_KEYS = ("chunk_id", "strategy", "doc", "lang", "kind", "provision_ids", "unit", "header", "text", "xrefs",
                "amended_by", "version")


def content_hash(chunk: dict) -> str:
    return hashlib.sha1(f"{chunk['header']}\n{chunk['text']}".encode("utf-8")).hexdigest()


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class NumpyStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.chunks: dict[str, dict] = {}
        self.vecs: dict[str, np.ndarray] = {}

    def upsert(self, chunks: list[dict], vectors: np.ndarray) -> None:
        for c, v in zip(chunks, vectors):
            self.chunks[c["chunk_id"]] = {**c, "hash": content_hash(c)}
            self.vecs[c["chunk_id"]] = np.asarray(v, dtype=np.float32)

    def delete(self, chunk_ids: list[str]) -> None:
        for cid in chunk_ids:
            self.chunks.pop(cid, None)
            self.vecs.pop(cid, None)

    def hashes(self, lang: str) -> dict[str, str]:
        return {cid: c["hash"] for cid, c in self.chunks.items() if c["lang"] == lang}

    def vectors(self, chunk_ids: list[str]) -> np.ndarray:
        return np.stack([self.vecs[c] for c in chunk_ids])

    def search(self, qvec: np.ndarray, lang: str, k: int, docs: list[str] | None = None) -> list[tuple[dict, float]]:
        ids = [cid for cid, c in self.chunks.items() if c["lang"] == lang and (docs is None or c["doc"] in docs)]
        if not ids:
            return []
        sims = self.vectors(ids) @ qvec
        top = np.argsort(-sims)[:k]
        return [(self.chunks[ids[i]], float(sims[i])) for i in top]


class QdrantStore:
    """Qdrant collection with payload indexes on lang and doc for filtered search."""

    def __init__(self, url: str | None = None, path: str | None = None, dim: int = 1024,
                 collection: str = COLLECTION):
        from qdrant_client import QdrantClient, models

        self.m = models
        self.client = QdrantClient(url=url) if url else QdrantClient(path=path or ":memory:")
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
            if url:  # payload indexes are a server feature; local mode filters without them
                for key in ("lang", "doc"):
                    self.client.create_payload_index(collection, key, models.PayloadSchemaType.KEYWORD)

    def upsert(self, chunks: list[dict], vectors: np.ndarray, batch: int = 128) -> None:
        for i in range(0, len(chunks), batch):
            pts = [self.m.PointStruct(id=point_id(c["chunk_id"]), vector=np.asarray(v, dtype=np.float32).tolist(),
                                      payload={**{k: c.get(k) for k in PAYLOAD_KEYS}, "hash": content_hash(c)})
                   for c, v in zip(chunks[i:i + batch], vectors[i:i + batch])]
            self.client.upsert(self.collection, pts)

    def delete(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self.client.delete(self.collection, self.m.PointIdsList(points=[point_id(c) for c in chunk_ids]))

    def _filter(self, lang: str, docs: list[str] | None):
        must = [self.m.FieldCondition(key="lang", match=self.m.MatchValue(value=lang))]
        if docs is not None:
            must.append(self.m.FieldCondition(key="doc", match=self.m.MatchAny(any=list(docs))))
        return self.m.Filter(must=must)

    def hashes(self, lang: str) -> dict[str, str]:
        out, offset = {}, None
        while True:
            pts, offset = self.client.scroll(self.collection, scroll_filter=self._filter(lang, None), limit=1000,
                                             offset=offset, with_payload=["chunk_id", "hash"], with_vectors=False)
            out.update({p.payload["chunk_id"]: p.payload["hash"] for p in pts})
            if offset is None:
                return out

    def vectors(self, chunk_ids: list[str]) -> np.ndarray:
        pts = self.client.retrieve(self.collection, [point_id(c) for c in chunk_ids], with_vectors=True,
                                   with_payload=["chunk_id"])
        by_id = {p.payload["chunk_id"]: p.vector for p in pts}
        return np.asarray([by_id[c] for c in chunk_ids], dtype=np.float32)

    def search(self, qvec: np.ndarray, lang: str, k: int, docs: list[str] | None = None) -> list[tuple[dict, float]]:
        res = self.client.query_points(self.collection, query=np.asarray(qvec, dtype=np.float32).tolist(),
                                       query_filter=self._filter(lang, docs), limit=k, with_payload=True)
        return [(dict(p.payload), float(p.score)) for p in res.points]
