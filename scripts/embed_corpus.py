"""Embed every chunk file with the chosen models and cache vectors in data/index/.

Usage: python scripts/embed_corpus.py [--models bge-m3 e5-small] [--device cuda]
Also embeds the benchmark questions (data/index/<model>/questions.npy).
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

from lexrag.retrieval.dense import EMBEDDERS, INDEX_DIR, Embedder, build_or_load  # noqa: E402

P = ROOT / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(EMBEDDERS))
    ap.add_argument("--strategies", nargs="+", default=["structural", "fixed", "article"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    qs = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").open(encoding="utf-8")]
    timings = {}
    for key in args.models:
        emb = Embedder(key, args.device)
        qv = emb.encode_queries([q["question"] for q in qs])
        (INDEX_DIR / key).mkdir(parents=True, exist_ok=True)
        np.save(INDEX_DIR / key / "questions.npy", qv.astype(np.float16))
        (INDEX_DIR / key / "questions.ids.json").write_text(json.dumps([q["id"] for q in qs]))
        for strategy in args.strategies:
            for lang in ("en", "sv"):
                chunks = [json.loads(line) for line in (P / f"chunks_{strategy}_{lang}.jsonl").open(encoding="utf-8")]
                t = time.time()
                vecs = build_or_load(chunks, key, strategy, lang, emb)
                dt = time.time() - t
                timings[f"{key}/{strategy}_{lang}"] = {"chunks": len(chunks), "seconds": round(dt, 1)}
                print(f"{key} {strategy}_{lang}: {vecs.shape} in {dt:.0f}s", flush=True)
    (INDEX_DIR / "embed_timings.json").write_text(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
