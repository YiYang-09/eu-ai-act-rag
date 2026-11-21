"""Graph expansion at query time.

Given the seed chunks from hybrid retrieval, add chunks one hop away in the
legal graph, then let the cross-encoder decide what makes the top k:

1. Term linking: defined terms named in the question ("deployer", "biometric data")
   add their definition, and the equivalent definition in the other act (SAME_AS).
2. References: provisions the top seeds cite (REFERS_TO), and passages that cite
   the seeds' provisions (reverse REFERS_TO). From each provision only the chunk
   closest to the question (dense similarity) is taken.
3. Dates: questions asking "when / from when / by when" pull in the act's
   entry-into-force article (APPLIES_FROM), which embeds poorly because it is a
   list of chapter numbers and dates.
"""

from __future__ import annotations

import re

import networkx as nx
import numpy as np

TEMPORAL = {
    "en": re.compile(r"\b(when|from when|by when|date|deadline|apply from|applicable|enter into force|in force)\b", re.I),
    "sv": re.compile(r"\b(när|från och med|senast|datum|tillämpas|träder i kraft|gäller från)\b", re.I),
}
# stricter: the question asks for an application date, not "when is X required" (a condition)
DATE_INTENT = {
    "en": re.compile(r"\b(apply|applies|applicable|application|in force|enter into force|from when|by when|date|deadline)\b", re.I),
    "sv": re.compile(r"\b(tillämpas|tillämpliga|gäller|träder i kraft|från och med|datum|senast)\b", re.I),
}


def wants_date(question: str, lang: str) -> bool:
    return bool(TEMPORAL[lang].search(question) and DATE_INTENT[lang].search(question))



class GraphExpander:
    def __init__(self, g: nx.MultiDiGraph, chunks: list[dict], vectors: np.ndarray,
                 seeds_to_expand: int = 10, refs_per_seed: int = 3, cited_by_per_seed: int = 2, cap: int = 20,
                 max_term_df: float = 0.10):
        self.g = g
        self.lang = g.graph["lang"]
        self.chunks = {c["chunk_id"]: c for c in chunks}
        self.row = {c["chunk_id"]: i for i, c in enumerate(chunks)}
        self.vectors = vectors
        self.k_seeds, self.k_refs, self.k_cited, self.cap = seeds_to_expand, refs_per_seed, cited_by_per_seed, cap
        texts = [c["text"].lower() for c in chunks]
        terms = []
        for n, d in g.nodes(data=True):
            if d.get("kind") != "term":
                continue
            df = sum(1 for t in texts if d["label"].lower() in t) / max(1, len(texts))
            if df <= max_term_df:  # "AI system", "personal data" appear everywhere and link to nothing useful
                terms.append((n, d["label"]))
        # longest labels first so "high-risk AI system" wins over "AI system"
        self.terms = sorted(terms, key=lambda t: -len(t[1]))

    def _sim(self, qvec: np.ndarray, cids: list[str]) -> list[tuple[str, float]]:
        cids = [c for c in cids if c in self.row]
        if not cids:
            return []
        if callable(self.vectors):  # e.g. a vector store that fetches vectors by chunk id
            mat = self.vectors(cids)
        else:
            mat = self.vectors[[self.row[c] for c in cids]]
        sims = mat @ qvec
        return sorted(zip(cids, sims.tolist()), key=lambda x: -x[1])

    def _best_chunks(self, qvec, provision: str, n: int = 1) -> list[str]:
        cids = self.g.graph["by_provision"].get(provision, [])
        return [c for c, _ in self._sim(qvec, cids)[:n]]

    def link_terms(self, question: str) -> list[str]:
        q = question.lower()
        found, taken = [], []
        for node, label in self.terms:
            m = re.search(rf"(?<![\w-]){re.escape(label.lower())}(s|es|en|na|er|erna)?(?![\w-])", q)
            if m and not any(a <= m.start() < b for a, b in taken):
                found.append(node)
                taken.append((m.start(), m.end()))
        return found

    def expand(self, question: str, qvec: np.ndarray, seeds: list[dict]) -> list[tuple[dict, str]]:
        seed_ids = {c["chunk_id"] for c in seeds}
        added: dict[str, str] = {}

        def add(cid: str, why: str):
            if cid in seed_ids or cid not in self.chunks:
                return
            if cid not in added or why.startswith(("definition", "entry into force")):
                added[cid] = why

        for term in self.link_terms(question):
            for t in [term] + [v for _, v, d in self.g.out_edges(term, data=True) if d["rel"] == "SAME_AS"]:
                for _, cid, d in self.g.out_edges(t, data=True):
                    if d["rel"] == "DEFINED_IN":
                        add(cid, f"definition of {self.g.nodes[t]['label']}")

        for seed in seeds[: self.k_seeds]:
            sid = seed["chunk_id"]
            refs = [v for _, v, d in self.g.out_edges(sid, data=True) if d["rel"] == "REFERS_TO"]
            best = [(p, self._best_chunks(qvec, p)) for p in refs]
            ranked = self._sim(qvec, [b[0] for _, b in best if b])
            for cid, _ in ranked[: self.k_refs]:
                add(cid, f"cited by {sid}")
            prov = seed["provision_ids"][0]
            citing = [u for u, _, d in self.g.in_edges(prov, data=True) if d["rel"] == "REFERS_TO"]
            for cid, _ in self._sim(qvec, citing)[: self.k_cited]:
                add(cid, f"cites {prov}")

        if TEMPORAL[self.lang].search(question):
            docs = {s["doc"] for s in seeds[: self.k_seeds]}
            for doc in docs:
                provs = {s["provision_ids"][0] for s in seeds if s["doc"] == doc}
                targets = {v for p in provs for _, v, d in self.g.out_edges(p, data=True) if d["rel"] == "APPLIES_FROM"}
                for cid, _ in self._sim(qvec, list(targets))[:2]:
                    add(cid, "entry into force / application dates")

        # definitions and date articles embed poorly, so they bypass the similarity cut
        priority = [cid for cid, why in added.items() if why.startswith(("definition", "entry into force"))]
        rest = [cid for cid, _ in self._sim(qvec, [c for c in added if c not in priority])]
        keep = priority + rest[: max(0, self.cap - len(priority))]
        return [(self.chunks[cid], added[cid]) for cid in keep]
