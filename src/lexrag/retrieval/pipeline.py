"""The production retrieval pipeline: BM25 + bge-m3 fused with RRF, then cross-encoder reranking."""

from __future__ import annotations

import json
from pathlib import Path

from .bm25 import BM25Index
from .dense import DenseIndex, Embedder, build_or_load
from .hybrid import rrf
from .rerank import Reranker, limit_recitals

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "data" / "processed"


class Retriever:
    def __init__(self, strategy: str = "structural", model: str = "bge-m3", device: str | None = None,
                 candidates: int = 50, pool: int = 30, graph: bool = False, max_recitals: int | None = None):
        self.chunks = {lang: [json.loads(x) for x in (P / f"chunks_{strategy}_{lang}.jsonl").open(encoding="utf-8")]
                       for lang in ("en", "sv")}
        self.embedder = Embedder(model, device)
        self.bm25 = {lang: BM25Index(c, lang) for lang, c in self.chunks.items()}
        self.dense = {lang: DenseIndex(c, build_or_load(c, model, strategy, lang, self.embedder))
                      for lang, c in self.chunks.items()}
        self.reranker = Reranker(device=device)
        self.candidates, self.pool = candidates, pool
        self.max_recitals = max_recitals
        self.expanders = None
        if graph:
            from ..graph.build import build_graph
            from ..graph.expand import GraphExpander

            self.expanders = {lang: GraphExpander(build_graph(c, lang), c, self.dense[lang].vectors)
                              for lang, c in self.chunks.items()}

    def search(self, query: str, lang: str, k: int = 5, query_id: str | None = None) -> list[tuple[dict, float]]:
        kw = [c for c, _ in self.bm25[lang].search(query, self.candidates)]
        qv = self.embedder.encode_queries([query])[0]
        dn = [c for c, _ in self.dense[lang].search_vec(qv, self.candidates)]
        fused = [c for c, _ in rrf([kw, dn], k=self.pool)]
        ranked = self.reranker.rerank(query, fused, k=len(fused), query_id=query_id)
        if self.expanders is not None:  # expand from the 10 best, then rerank the union
            extra = [c for c, _ in self.expanders[lang].expand(query, qv, [c for c, _ in ranked[:10]])]
            ranked = self.reranker.rerank(query, fused + extra, k=len(fused) + len(extra), query_id=query_id)
        scores = {c["chunk_id"]: s for c, s in ranked}
        return [(c, scores[c["chunk_id"]]) for c in limit_recitals([c for c, _ in ranked], k, self.max_recitals)]
