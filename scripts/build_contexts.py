"""Retrieve the top-k chunks for every benchmark question and freeze them to a file.

Generation then runs from this file, so the LLM step needs no embedding models
(it can run on a machine that only has Ollama) and every model sees exactly the
same evidence. Swedish questions retrieve from the Swedish index.

Usage: python scripts/build_contexts.py [--k 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexrag.retrieval.pipeline import Retriever  # noqa: E402

KEEP = ("chunk_id", "doc", "lang", "kind", "provision_ids", "unit", "header", "text", "amended_by")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(ROOT / "eval" / "contexts" / "hybrid_rerank_top5.jsonl"))
    args = ap.parse_args()
    qs = [json.loads(x) for x in (ROOT / "eval" / "questions.jsonl").open(encoding="utf-8")]
    r = Retriever(device=args.device)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, q in enumerate(qs, 1):
            hits = r.search(q["question"], q["lang"], k=args.k, query_id=q["id"])
            row = {"id": q["id"], "lang": q["lang"],
                   "contexts": [{**{k: c[k] for k in KEEP}, "score": round(s, 4)} for c, s in hits]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 10 == 0:
                r.reranker.save()
                print(f"{i}/{len(qs)}", flush=True)
    r.reranker.save()


if __name__ == "__main__":
    main()
