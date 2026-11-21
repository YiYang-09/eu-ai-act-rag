"""GraphRAG round 2: a graph built from LLM-extracted triples (scripts/extract_triples.py).

Nodes are normalised entity strings; every triple links its subject and object and remembers the
chunk it came from. Questions are linked to entities by embedding similarity (bge-m3 is
multilingual, so Swedish questions link to the English entity names), not by string match.

Expansion, for a question with seeds (the reranked top 10):
  1. linked entities (top n by cosine, above a threshold) -> chunks whose triples mention them
  2. one hop along triples from linked entities -> chunks of the neighbouring triples
  3. APPLIES_FROM / PENALTY triples of linked entities are priority (dates and fines embed poorly)
Candidates are ranked by similarity to the question and capped, like the rule-based expander.
The Swedish graph reuses the English triples through the aligned chunk ids.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

_ART = re.compile(r"^(the|a|an|any|each|all)\s+")


def norm_entity(s: str) -> str:
    s = _ART.sub("", s.strip().lower())
    s = re.sub(r"[‘’'\"`]", "", s)
    s = re.sub(r"\s+", " ", s)
    if len(s) > 4 and s.endswith("s") and not s.endswith(("ss", "us", "is")):
        s = s[:-1]  # crude singular: "deployers" -> "deployer"
    return s


def load_triples(path: Path) -> dict[str, list[dict]]:
    out = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        out[r["chunk_id"]] = [{"s": norm_entity(t["s"]), "r": t["r"], "o": norm_entity(t["o"])} for t in r["triples"]]
    return out


class LLMGraph:
    def __init__(self, triples_by_chunk: dict[str, list[dict]]):
        self.triples = triples_by_chunk
        self.mentions: dict[str, set[str]] = defaultdict(set)  # entity -> chunk ids
        self.edges: dict[str, list[tuple[str, str, str]]] = defaultdict(list)  # entity -> (rel, other, chunk)
        for cid, ts in triples_by_chunk.items():
            for t in ts:
                self.mentions[t["s"]].add(cid)
                self.mentions[t["o"]].add(cid)
                self.edges[t["s"]].append((t["r"], t["o"], cid))
                self.edges[t["o"]].append((t["r"], t["s"], cid))
        self.entities = sorted(self.mentions)

    def stats(self) -> dict:
        n_t = sum(len(ts) for ts in self.triples.values())
        rels = defaultdict(int)
        for ts in self.triples.values():
            for t in ts:
                rels[t["r"]] += 1
        return {"chunks": len(self.triples), "chunks_with_triples": sum(1 for ts in self.triples.values() if ts),
                "triples": n_t, "entities": len(self.entities), "relations": dict(sorted(rels.items()))}


class LLMGraphExpander:
    def __init__(self, graph: LLMGraph, chunks: list[dict], vectors, entity_vecs: np.ndarray,
                 link_top: int = 5, link_min: float = 0.55, max_entity_df: int = 40,
                 seeds_to_expand: int = 10, cap: int = 20):
        self.g = graph
        self.lang = chunks[0]["lang"]
        self.chunks = {c["chunk_id"]: c for c in chunks}
        self.row = {c["chunk_id"]: i for i, c in enumerate(chunks)}
        self.vectors = vectors
        self.entity_vecs = entity_vecs  # rows aligned with graph.entities
        self.link_top, self.link_min, self.max_df = link_top, link_min, max_entity_df
        self.k_seeds, self.cap = seeds_to_expand, cap

    def _local(self, en_cid: str) -> str:
        return en_cid if self.lang == "en" else "sv:" + en_cid.split(":", 1)[1]

    def _en(self, cid: str) -> str:
        return "en:" + cid.split(":", 1)[1]

    def _sim(self, qvec, cids):
        cids = [c for c in cids if c in self.row]
        if not cids:
            return []
        mat = self.vectors(cids) if callable(self.vectors) else self.vectors[[self.row[c] for c in cids]]
        return sorted(zip(cids, (mat @ qvec).tolist()), key=lambda x: -x[1])

    def link(self, qvec: np.ndarray) -> list[tuple[str, float]]:
        sims = self.entity_vecs @ qvec
        out = []
        for i in np.argsort(-sims):
            if sims[i] < self.link_min or len(out) >= self.link_top:
                break
            e = self.g.entities[i]
            if len(self.g.mentions[e]) <= self.max_df:  # "provider", "ai system": everywhere, no signal
                out.append((e, float(sims[i])))
        return out

    def expand(self, question: str, qvec: np.ndarray, seeds: list[dict]) -> list[tuple[dict, str]]:
        seed_ids = {c["chunk_id"] for c in seeds}
        added: dict[str, str] = {}
        priority: list[str] = []

        def add(en_cid: str, why: str, prio: bool = False):
            cid = self._local(en_cid)
            if cid in seed_ids or cid not in self.chunks:
                return
            if prio and cid not in priority:
                priority.append(cid)
            added.setdefault(cid, why)

        linked = self.link(qvec)
        for e, _ in linked:
            for cid in self.g.mentions[e]:
                add(cid, f"mentions '{e}'")
            for rel, other, cid in self.g.edges[e]:
                add(cid, f"'{e}' {rel} '{other}'", prio=rel in ("APPLIES_FROM", "PENALTY"))
        # entities of the seeds: what else is said about the actors and obligations in the best chunks
        for seed in seeds[: self.k_seeds]:
            for t in self.g.triples.get(self._en(seed["chunk_id"]), []):
                for e in (t["s"], t["o"]):
                    if len(self.g.mentions[e]) <= self.max_df:
                        for cid in self.g.mentions[e]:
                            add(cid, f"shares '{e}' with {seed['chunk_id']}")
        rest = [cid for cid, _ in self._sim(qvec, [c for c in added if c not in priority])]
        prio = [cid for cid, _ in self._sim(qvec, priority)][: self.cap // 2]
        keep = prio + rest[: max(0, self.cap - len(prio))]
        return [(self.chunks[cid], added[cid]) for cid in keep]


class CombinedExpander:
    """Union of the rule-based and the LLM expander (each capped), for the 'both' configuration."""

    def __init__(self, *expanders):
        self.expanders = expanders

    def expand(self, question, qvec, seeds):
        seen, out = set(), []
        for ex in self.expanders:
            for c, why in ex.expand(question, qvec, seeds):
                if c["chunk_id"] not in seen:
                    seen.add(c["chunk_id"])
                    out.append((c, why))
        return out
