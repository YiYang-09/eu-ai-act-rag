"""The question-answering service used by the API.

Pipeline per request:
  BM25 (in memory) + dense search (vector store, filtered by language and the
  caller's allowed documents) -> RRF -> cross-encoder rerank -> graph expansion
  from the top 10 -> rerank -> at most 1 recital in the top k -> LLM with
  numbered sources -> answer + citations linked to EUR-Lex.

All heavy parts (embedder, reranker, LLM, store) are injected, so tests run the
whole pipeline with small fakes.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..corpus.sources import CURRENT
from ..generation.prompt import build_messages, cite_label
from ..graph.build import build_graph
from ..graph.expand import GraphExpander, wants_date
from ..retrieval.bm25 import BM25Index
from ..retrieval.hybrid import rrf
from ..retrieval.rerank import limit_recitals, pin_chunks
from .tracing import Tracer

ROOT = Path(__file__).resolve().parents[3]
CELEX = {s.key: s.celex for s in CURRENT if s.role == "provisions"}
CELEX_RECITALS = {s.key: s.celex for s in CURRENT if s.role == "recitals"}
SV_HINT = re.compile(r"[åäö]|\b(och|att|vad|när|hur|får|måste|ska|enligt|är)\b", re.I)


def detect_lang(text: str) -> str:
    return "sv" if SV_HINT.search(text) else "en"


def eurlex_url(provision_id: str, lang: str) -> str:
    doc, _, local = provision_id.partition(":")
    celex = CELEX_RECITALS[doc] if local.startswith("rct_") else CELEX[doc]
    return f"https://eur-lex.europa.eu/legal-content/{lang.upper()}/TXT/?uri=CELEX:{celex}#{local}"


def parse_json_reply(text: str) -> dict:
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        d = json.loads(m.group() if m else text)
        return {"answer": str(d.get("answer", "")).strip(),
                "sources": [int(s) for s in d.get("sources", []) if str(s).strip().isdigit()],
                "insufficient": bool(d.get("insufficient", False))}
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        return {"answer": text.strip(), "sources": [int(x) for x in re.findall(r"\[(\d)\]", text)],
                "insufficient": False}


@dataclass
class Citation:
    n: int
    label: str
    provision_id: str
    chunk_id: str
    url: str
    header: str
    text: str
    kind: str
    cited: bool
    rerank_score: float
    via: str = "retrieval"


@dataclass
class Answer:
    question: str
    lang: str
    role: str
    answer: str
    insufficient: bool
    citations: list[Citation]
    model: str
    timings_ms: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)


class QAService:
    def __init__(self, chunks: list[dict], store, embedder, reranker, llm, access: dict[str, list[str]],
                 k: int = 5, candidates: int = 50, pool: int = 30, max_recitals: int | None = 1,
                 use_graph: bool = True, tracer: Tracer | None = None):
        self.chunks = {lang: [c for c in chunks if c["lang"] == lang] for lang in ("en", "sv")}
        self.store, self.embedder, self.reranker, self.llm = store, embedder, reranker, llm
        self.access = access
        self.k, self.candidates, self.pool, self.max_recitals = k, candidates, pool, max_recitals
        self.tracer = tracer or Tracer()
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.bm25 = {lang: BM25Index(cs, lang) for lang, cs in self.chunks.items() if cs}
        self.expanders = {}
        if use_graph:
            for lang, cs in self.chunks.items():
                if cs:
                    self.expanders[lang] = GraphExpander(build_graph(cs, lang), cs, store.vectors)

    # -- retrieval -------------------------------------------------------------------------------
    def retrieve(self, question: str, lang: str, role: str, trace=None) -> tuple[list[tuple[dict, float, str]], dict]:
        docs = self.access.get(role)
        if docs is None:
            raise PermissionError(f"unknown role {role!r}")
        t = {}
        t0 = time.perf_counter()
        qv = self.embedder.encode_queries([question])[0]
        allowed = lambda c: c["doc"] in docs  # noqa: E731
        with self.tracer.span(trace, "retrieve", input={"lang": lang, "role": role}):
            kw = [c for c, _ in self.bm25[lang].search(question, self.candidates * 2) if allowed(c)][: self.candidates]
            # map store hits back to the in-memory chunk records, so BM25 and dense results are the same objects
            dn = [self.by_id.get(c["chunk_id"], c) for c, _ in self.store.search(qv, lang, self.candidates, docs)]
            fused = [c for c, _ in rrf([kw, dn], k=self.pool)]
        t["retrieve"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        with self.tracer.span(trace, "rerank"):
            ranked = self.reranker.rerank(question, fused, k=len(fused))
            via = {c["chunk_id"]: "retrieval" for c in fused}
            pins: list[str] = []
            if lang in self.expanders:
                extra = [(c, why) for c, why in self.expanders[lang].expand(question, qv, [c for c, _ in ranked[:10]])
                         if allowed(c)]
                via.update({c["chunk_id"]: f"graph: {why}" for c, why in extra})
                if wants_date(question, lang):
                    pins = [c["chunk_id"] for c, why in extra if why.startswith("entry into force")][:1]
                ranked = self.reranker.rerank(question, fused + [c for c, _ in extra], k=len(fused) + len(extra))
        t["rerank_graph"] = time.perf_counter() - t0
        scores = {c["chunk_id"]: s for c, s in ranked}
        order = [c for c, _ in ranked]
        top = pin_chunks(limit_recitals(order, self.k, self.max_recitals), order, pins, self.k)
        return [(c, scores[c["chunk_id"]], via.get(c["chunk_id"], "retrieval")) for c in top], t

    # -- answering -------------------------------------------------------------------------------
    def ask(self, question: str, lang: str | None = None, role: str = "public") -> Answer:
        t_start = time.perf_counter()
        lang = lang or detect_lang(question)
        trace = self.tracer.trace("ask", input={"question": question, "lang": lang, "role": role})
        hits, timings = self.retrieve(question, lang, role, trace)
        contexts = [c for c, _, _ in hits]
        t0 = time.perf_counter()
        msgs = build_messages(question, contexts)
        out = self.llm.chat(msgs)
        timings["generate"] = time.perf_counter() - t0
        reply = parse_json_reply(out["content"])
        timings["total"] = time.perf_counter() - t_start
        self.tracer.generation(trace, name="answer", model=getattr(self.llm, "model", "?"), input=msgs,
                               output=out["content"],
                               usage={"input": out.get("prompt_tokens"), "output": out.get("output_tokens")})
        cited = set(reply["sources"])
        citations = [Citation(n=i, label=cite_label(c), provision_id=c["provision_ids"][0], chunk_id=c["chunk_id"],
                              url=eurlex_url(c["provision_ids"][0], lang), header=c["header"], text=c["text"],
                              kind=c["kind"], cited=i in cited, rerank_score=round(s, 4), via=via)
                     for i, (c, s, via) in enumerate(hits, 1)]
        ans = Answer(question=question, lang=lang, role=role, answer=reply["answer"],
                     insufficient=reply["insufficient"], citations=citations,
                     model=getattr(self.llm, "model", "?"),
                     timings_ms={k: round(v * 1000) for k, v in timings.items()},
                     tokens={"prompt": out.get("prompt_tokens"), "output": out.get("output_tokens")})
        if trace is not None:
            trace.update(output={"answer": ans.answer, "insufficient": ans.insufficient,
                                 "cited": [c.label for c in citations if c.cited]})
        return ans


def load_chunks(strategy: str = "structural") -> list[dict]:
    out = []
    for lang in ("en", "sv"):
        path = ROOT / "data" / "processed" / f"chunks_{strategy}_{lang}.jsonl"
        out += [json.loads(x) for x in path.open(encoding="utf-8")]
    return out


def load_access(path: Path | None = None) -> dict[str, list[str]]:
    d = json.loads((path or ROOT / "configs" / "access.json").read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_")}
