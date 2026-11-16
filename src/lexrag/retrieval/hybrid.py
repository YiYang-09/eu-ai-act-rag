"""Hybrid retrieval: reciprocal rank fusion (RRF) of keyword and dense rankings."""

from __future__ import annotations


def rrf(rankings: list[list[dict]], k: int = 10, c: int = 60) -> list[tuple[dict, float]]:
    """Fuse ranked chunk lists. score(d) = sum over rankings of 1 / (c + rank)."""
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, 1):
            cid = chunk["chunk_id"]
            by_id[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (c + rank)
    top = sorted(scores, key=lambda cid: -scores[cid])[:k]
    return [(by_id[cid], scores[cid]) for cid in top]
