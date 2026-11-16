"""Dense retrieval: embed chunks once, cache the vectors, search by cosine similarity.

Vectors are cached as float16 .npy files under data/index/<model>/, next to a
JSON list of chunk ids, and are reused when the chunk ids still match. Search
is exact (a matrix product); at ~1,500 chunks per index an ANN index would
only add noise to the evaluation. The serving stack uses Qdrant instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
INDEX_DIR = ROOT / "data" / "index"


@dataclass(frozen=True)
class EmbedderSpec:
    key: str
    hf_name: str
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int = 512


EMBEDDERS = {
    "bge-m3": EmbedderSpec("bge-m3", "BAAI/bge-m3", max_seq_length=2048),
    "e5-small": EmbedderSpec("e5-small", "intfloat/multilingual-e5-small", "query: ", "passage: ", 512),
}


def chunk_text(c: dict) -> str:
    return f"{c['header']}\n{c['text']}" if c.get("header") else c["text"]


class Embedder:
    def __init__(self, key: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.spec = EMBEDDERS[key]
        self.model = SentenceTransformer(self.spec.hf_name, device=device)
        self.model.max_seq_length = self.spec.max_seq_length

    def encode_passages(self, texts: list[str], batch_size: int = 8, progress: bool = False) -> np.ndarray:
        return self.model.encode([self.spec.passage_prefix + t for t in texts], batch_size=batch_size,
                                 normalize_embeddings=True, show_progress_bar=progress)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.model.encode([self.spec.query_prefix + t for t in texts], batch_size=16,
                                 normalize_embeddings=True, show_progress_bar=False)


def index_paths(model_key: str, strategy: str, lang: str) -> tuple[Path, Path]:
    d = INDEX_DIR / model_key
    return d / f"{strategy}_{lang}.npy", d / f"{strategy}_{lang}.ids.json"


def build_or_load(chunks: list[dict], model_key: str, strategy: str, lang: str,
                  embedder: Embedder | None = None) -> np.ndarray:
    """Return the (n_chunks, dim) matrix for these chunks, embedding them if not cached."""
    vec_path, ids_path = index_paths(model_key, strategy, lang)
    ids = [c["chunk_id"] for c in chunks]
    if vec_path.exists() and ids_path.exists() and json.loads(ids_path.read_text()) == ids:
        return np.load(vec_path).astype(np.float32)
    if embedder is None:
        raise FileNotFoundError(f"no cached vectors for {model_key}/{strategy}_{lang}; run scripts/embed_corpus.py")
    vecs = embedder.encode_passages([chunk_text(c) for c in chunks], progress=True)
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, vecs.astype(np.float16))
    ids_path.write_text(json.dumps(ids))
    return vecs.astype(np.float32)


class DenseIndex:
    def __init__(self, chunks: list[dict], vectors: np.ndarray):
        assert len(chunks) == len(vectors)
        self.chunks = chunks
        self.vectors = vectors

    def search_vec(self, qvec: np.ndarray, k: int = 10) -> list[tuple[dict, float]]:
        scores = self.vectors @ qvec.astype(np.float32)
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]
