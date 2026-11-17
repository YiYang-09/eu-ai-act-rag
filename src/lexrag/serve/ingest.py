"""Incremental ingestion: embed and upsert only chunks whose content changed.

Each stored chunk carries a SHA-1 of its header + text. On re-ingestion the
new chunk set is diffed against the stored hashes:
  new or changed -> embed + upsert,  unchanged -> skip,  gone -> delete.
When the AI Act is amended, only the amended paragraphs are re-embedded.
"""

from __future__ import annotations

from dataclasses import dataclass

from .store import content_hash


@dataclass
class IngestReport:
    lang: str
    total: int
    added: int
    changed: int
    unchanged: int
    deleted: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def ingest(store, chunks: list[dict], embed_fn, lang: str) -> IngestReport:
    """embed_fn(list of chunks) -> np.ndarray of normalised vectors."""
    stored = store.hashes(lang)
    current = {c["chunk_id"]: c for c in chunks if c["lang"] == lang}
    added = [c for cid, c in current.items() if cid not in stored]
    changed = [c for cid, c in current.items() if cid in stored and stored[cid] != content_hash(c)]
    gone = [cid for cid in stored if cid not in current]
    todo = added + changed
    if todo:
        store.upsert(todo, embed_fn(todo))
    store.delete(gone)
    return IngestReport(lang, len(current), len(added), len(changed),
                        len(current) - len(todo), len(gone))
