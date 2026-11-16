"""BM25 keyword retrieval over chunk records (the lexical half of hybrid search)."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

STOP = {
    "en": set("the of and to a in for or be by that is shall on with as any an are which this its it at from where "
              "under such their not have has other may been if than who what when how does do can i".split()),
    "sv": set("och att i en ett som för av på den det de är med till om ska eller har inte vid från enligt under "
              "så sådan sådana dessa denna detta vad när hur får kan måste".split()),
}
TOKEN = re.compile(r"[0-9a-zåäöéü]+(?:[-'][0-9a-zåäöéü]+)*", re.IGNORECASE)


def tokenize(text: str, lang: str) -> list[str]:
    return [t for t in (m.group().lower() for m in TOKEN.finditer(text)) if t not in STOP[lang]]


class BM25Index:
    def __init__(self, chunks: list[dict], lang: str):
        self.chunks = chunks
        self.lang = lang
        docs = [tokenize(f"{c['header']}\n{c['text']}", lang) for c in chunks]
        self.bm25 = BM25Okapi(docs)

    def search(self, query: str, k: int = 10) -> list[tuple[dict, float]]:
        scores = self.bm25.get_scores(tokenize(query, self.lang))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]
