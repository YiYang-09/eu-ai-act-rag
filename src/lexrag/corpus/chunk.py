"""Chunking strategies compared in the ablation.

* ``structural`` (default): one chunk per paragraph of an article, per
  top-level point of an annex, per recital. Paragraphs longer than
  ``max_words`` are split at their list points, with the paragraph's lead-in
  sentence repeated in each part so "the following definitions apply:" style
  context is not lost. Every chunk carries a breadcrumb header.
* ``article``: one chunk per article / annex / recital, no splitting.
* ``fixed``: sliding windows of ``window`` words over the running text of each
  document, ignoring structure (the usual "chat with PDF" baseline). Each
  window records which provisions it overlaps so it can still be scored.

All strategies emit the same record shape, so retrieval and evaluation code
does not care which one produced a chunk.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .parse import Provision, Unit
from .xref import extract_xrefs

DOC_NAMES = {
    "en": {"AIA": "AI Act (Regulation (EU) 2024/1689)", "GDPR": "GDPR (Regulation (EU) 2016/679)"},
    "sv": {"AIA": "AI-förordningen (förordning (EU) 2024/1689)",
           "GDPR": "Dataskyddsförordningen (GDPR, förordning (EU) 2016/679)"},
}
LABELS = {
    "en": {"article": "Article", "annex": "Annex", "recital": "Recital", "para": "paragraph"},
    "sv": {"article": "Artikel", "annex": "Bilaga", "recital": "Skäl", "para": "punkt"},
}


@dataclass
class Chunk:
    chunk_id: str
    strategy: str
    doc: str
    lang: str
    kind: str
    provision_ids: list[str]  # provisions this chunk belongs to / overlaps
    unit: str | None  # paragraph or point number within the provision
    header: str  # breadcrumb, e.g. "AI Act > Chapter III ... > Article 6 ... > paragraph 3"
    text: str
    xrefs: list[str] = field(default_factory=list)
    amended_by: list[str] = field(default_factory=list)
    version: str = ""

    @property
    def n_words(self) -> int:
        return len(self.text.split())

    def embed_text(self) -> str:
        """What gets embedded / BM25-indexed: header + body."""
        return f"{self.header}\n{self.text}" if self.header else self.text

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_words"] = self.n_words
        return d


def provision_header(p: Provision) -> str:
    L = LABELS[p.lang]
    parts = [DOC_NAMES[p.lang].get(p.doc, p.doc)]
    if p.chapter:
        parts.append(p.chapter)
    if p.section:
        parts.append(p.section)
    head = f"{L[p.kind]} {p.num}"
    if p.title:
        head += f" – {p.title}"
    parts.append(head)
    return " > ".join(parts)


def _words(s: str) -> int:
    return len(s.split())


def _pack_sentences(text: str, max_words: int, prefix: str = "part") -> list[tuple[str, str]]:
    sents = re.split(r"(?<=[.;:])\s+(?=[A-ZÅÄÖ(])", text)
    pieces, buf, size, k = [], [], 0, 1
    for s in sents:
        w = _words(s)
        if buf and size + w > max_words:
            pieces.append((f"{prefix} {k}".strip(), " ".join(buf)))
            buf, size, k = [], 0, k + 1
        buf.append(s)
        size += w
    if buf:
        pieces.append((f"{prefix} {k}".strip(), " ".join(buf)))
    return pieces


def _split_unit(unit: Unit, max_words: int) -> list[tuple[str, str]]:
    """Split a long unit into (part label, text) pieces at list points or sentences."""
    if _words(unit.text) <= max_words:
        return [("", unit.text)]
    if unit.points:
        pieces = []
        for label, text in _split_points(unit, max_words):
            if _words(text) > int(max_words * 1.3):  # a single huge point: fall back to sentences
                pieces += _pack_sentences(text, max_words, prefix=f"{label} part")
            else:
                pieces.append((label, text))
        return pieces
    return _pack_sentences(unit.text, max_words)


def _split_points(unit: Unit, max_words: int) -> list[tuple[str, str]]:
    """Group consecutive list points up to max_words, repeating the lead-in text."""
    def label(labels: list[str]) -> str:
        return f"{labels[0]}–{labels[-1]}" if len(labels) > 1 else labels[0]

    lead = unit.lead
    pieces: list[tuple[str, str]] = []
    group: list[str] = []
    labels: list[str] = []
    size = _words(lead)
    for lab, txt in unit.points:
        item = f"{lab} {txt}"
        if group and size + _words(item) > max_words:
            pieces.append((label(labels), "\n".join(([lead] if lead else []) + group)))
            group, labels, size = [], [], _words(lead)
        group.append(item)
        labels.append(lab)
        size += _words(item)
    if group:
        pieces.append((label(labels), "\n".join(([lead] if lead else []) + group)))
    return pieces


def _unit_key(p: Provision, idx: int, unit: Unit) -> str:
    """Stable unit id: paragraph number for articles, position for annexes."""
    if p.kind == "article" and unit.num and len(p.units) > 1:
        return unit.num
    return str(idx)


def chunk_structural(provs: list[Provision], max_words: int = 350) -> list[Chunk]:
    out = []
    for p in provs:
        base = provision_header(p)
        L = LABELS[p.lang]
        for idx, unit in enumerate(p.units):
            key = _unit_key(p, idx, unit)
            for part, text in _split_unit(unit, max_words):
                header = base
                if p.kind == "article" and len(p.units) > 1 and unit.num:
                    header += f" > {L['para']} {unit.num}"
                elif p.kind == "annex" and unit.num:
                    header += f" > {unit.num}"
                if part:
                    header += f" [{part}]"
                cid = f"{p.lang}:{p.pid}:{key}" + (f":{part}" if part else "")
                out.append(Chunk(
                    chunk_id=cid, strategy="structural", doc=p.doc, lang=p.lang, kind=p.kind,
                    provision_ids=[p.pid], unit=unit.num, header=header, text=text,
                    xrefs=extract_xrefs(text, p.lang, p.doc, p.pid),
                    amended_by=unit.amended_by, version=p.version,
                ))
    return out


def chunk_article(provs: list[Provision]) -> list[Chunk]:
    out = []
    for p in provs:
        text = p.text
        out.append(Chunk(
            chunk_id=f"{p.lang}:{p.pid}", strategy="article", doc=p.doc, lang=p.lang, kind=p.kind,
            provision_ids=[p.pid], unit=None, header=provision_header(p), text=text,
            xrefs=extract_xrefs(text, p.lang, p.doc, p.pid), amended_by=p.amended_by, version=p.version,
        ))
    return out


def chunk_fixed(provs: list[Provision], window: int = 200, overlap: int = 40) -> list[Chunk]:
    """Sliding word windows over each (doc, lang, kind-group) stream, no headers."""
    out = []
    streams: dict[tuple[str, str, str], list[tuple[str, str]]] = {}
    for p in provs:
        group = "recital" if p.kind == "recital" else "body"
        # keep the visible "Article 6" heading in the stream, as a PDF would
        L = LABELS[p.lang]
        head = f"{L[p.kind]} {p.num}" + (f" {p.title}" if p.title else "")
        streams.setdefault((p.doc, p.lang, group), []).append((p.pid, f"{head}\n{p.text}"))
    step = window - overlap
    for (doc, lang, group), items in streams.items():
        words: list[str] = []
        owners: list[str] = []
        for pid, text in items:
            ws = text.split()
            words.extend(ws)
            owners.extend([pid] * len(ws))
        version = next((p.version for p in provs if p.doc == doc and p.lang == lang), "")
        for k, start in enumerate(range(0, max(len(words) - overlap, 1), step)):
            span = slice(start, start + window)
            pids = list(dict.fromkeys(owners[span]))
            text = " ".join(words[span])
            out.append(Chunk(
                chunk_id=f"{lang}:{doc}:{group}:w{k:04d}", strategy="fixed", doc=doc, lang=lang,
                kind="recital" if group == "recital" else "mixed", provision_ids=pids, unit=None,
                header="", text=text, xrefs=extract_xrefs(text, lang, doc), version=version,
            ))
    return out


STRATEGIES = {"structural": chunk_structural, "article": chunk_article, "fixed": chunk_fixed}
