"""Load the corpus into the vector store, incrementally.

  python scripts/ingest.py --version 2026                    # current law (default)
  python scripts/ingest.py --version 2024                    # AI Act as published in 2024
  python scripts/ingest.py --demo                            # 2024 -> 2026 re-index, writes eval/results/ingest_demo.json

Store: --qdrant-url http://localhost:6333 (server), --qdrant-path data/qdrant (local file mode),
or --numpy (in memory, only useful with --demo).

Embedding uses bge-m3. Vectors already computed for the evaluation (data/index/bge-m3/*.npy)
are reused when a chunk's content hash matches, so the 2026 load needs no model on a fresh
machine that has the cache; everything else is embedded on the fly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexrag.corpus.chunk import chunk_structural  # noqa: E402
from lexrag.corpus.parse import Provision  # noqa: E402
from lexrag.serve.ingest import ingest  # noqa: E402
from lexrag.serve.store import NumpyStore, QdrantStore, content_hash  # noqa: E402

PROC = ROOT / "data" / "processed"
LANGS = ("en", "sv")


def read_provisions(name: str) -> list[Provision]:
    return [Provision.from_dict(json.loads(x)) for x in (PROC / f"{name}.jsonl").open(encoding="utf-8")]


def chunks_for(version: str, lang: str) -> list[dict]:
    if version == "2026":
        return [json.loads(x) for x in (PROC / f"chunks_structural_{lang}.jsonl").open(encoding="utf-8")]
    current = read_provisions(f"provisions_{lang}")
    original = read_provisions(f"aia_original_provisions_{lang}")
    # 2024: original AI Act articles + annexes; recitals and GDPR are the same in both versions
    provs = original + [p for p in current if not (p.doc == "AIA" and p.kind in ("article", "annex"))]
    return [c.to_dict() for c in chunk_structural(provs)]


class CachedEmbedder:
    """Reuses evaluation vectors by content hash; embeds the rest with bge-m3."""

    def __init__(self, device: str | None):
        self.device, self.model, self.cache, self.computed = device, None, {}, 0
        for lang in LANGS:
            vec = ROOT / "data" / "index" / "bge-m3" / f"structural_{lang}.npy"
            if vec.exists():
                chunks = chunks_for("2026", lang)
                for c, v in zip(chunks, np.load(vec)):
                    self.cache[content_hash(c)] = v

    def __call__(self, chunks: list[dict]) -> np.ndarray:
        out, todo = [None] * len(chunks), []
        for i, c in enumerate(chunks):
            v = self.cache.get(content_hash(c))
            if v is None:
                todo.append(i)
            else:
                out[i] = v
        if todo:
            if self.model is None:
                from lexrag.retrieval.dense import Embedder

                self.model = Embedder("bge-m3", self.device)
            from lexrag.retrieval.dense import chunk_text

            vecs = self.model.encode_passages([chunk_text(chunks[i]) for i in todo])
            for i, v in zip(todo, vecs):
                out[i] = v
                self.cache[content_hash(chunks[i])] = v
            self.computed += len(todo)
        return np.stack(out)


def load(store, version: str, embed) -> list[dict]:
    reports = []
    for lang in LANGS:
        t0 = time.perf_counter()
        r = ingest(store, chunks_for(version, lang), embed, lang).as_dict()
        r.update(version=version, seconds=round(time.perf_counter() - t0, 1))
        print(json.dumps(r))
        reports.append(r)
    return reports


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["2024", "2026"], default="2026")
    ap.add_argument("--demo", action="store_true", help="load 2024, then re-ingest 2026 and report the diff")
    ap.add_argument("--qdrant-url")
    ap.add_argument("--qdrant-path")
    ap.add_argument("--numpy", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args()

    if args.numpy:
        store = NumpyStore(1024)
    elif args.qdrant_url or args.qdrant_path:
        store = QdrantStore(url=args.qdrant_url, path=args.qdrant_path)
    else:
        store = QdrantStore(url="http://localhost:6333")
    embed = CachedEmbedder(args.device)

    if not args.demo:
        load(store, args.version, embed)
        return
    steps = load(store, "2024", embed) + load(store, "2026", embed)
    changed = {lang: [c for c in chunks_for("2026", lang) if c["doc"] == "AIA" and c["amended_by"]] for lang in LANGS}
    out = {"steps": steps, "chunks_marked_amended_2026": {k: len(v) for k, v in changed.items()},
           "note": "Second load re-embeds only added/changed chunks; unchanged chunks keep their vectors."}
    path = ROOT / "eval" / "results" / "ingest_demo.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
