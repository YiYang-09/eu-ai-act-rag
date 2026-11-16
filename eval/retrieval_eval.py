"""Retrieval evaluation: does the top-k contain the gold provisions?

Metrics per configuration (answerable questions only):
  hit@k     at least one gold provision among the top-k chunks
  recall@k  share of gold provisions covered by the top-k chunks (averaged)
  mrr@k     reciprocal rank of the first chunk that touches a gold provision

Usage: python eval/retrieval_eval.py [--k 5] [--retriever bm25]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexrag.retrieval.bm25 import BM25Index  # noqa: E402

P = ROOT / "data" / "processed"


def load_chunks(strategy: str, lang: str) -> list[dict]:
    return [json.loads(line) for line in (P / f"chunks_{strategy}_{lang}.jsonl").open(encoding="utf-8")]


def score(results: list[dict], gold: list[str], k: int) -> tuple[float, float, float]:
    gold_set = set(gold)
    covered: set[str] = set()
    rr = 0.0
    for rank, c in enumerate(results[:k], 1):
        hit = gold_set & set(c["provision_ids"])
        if hit and rr == 0.0:
            rr = 1.0 / rank
        covered |= hit
    return float(bool(covered)), len(covered) / len(gold_set), rr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "retrieval_bm25.json"))
    ap.add_argument("--gate", action="append", default=[],
                    help="strategy:lang:metric:min, e.g. structural:en:hit@5:0.70; exit 1 if below")
    args = ap.parse_args()
    qs = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").open(encoding="utf-8")]
    qs = [q for q in qs if q["answerable"]]
    rows = []
    for strategy in ("structural", "article", "fixed"):
        idx = {lang: BM25Index(load_chunks(strategy, lang), lang) for lang in ("en", "sv")}
        for q_lang in ("en", "sv"):
            subset = [q for q in qs if q["lang"] == q_lang]
            # Swedish questions are run both against the Swedish index and the English one
            for index_lang in (["en"] if q_lang == "en" else ["sv", "en"]):
                agg = [score([c for c, _ in idx[index_lang].search(q["question"], args.k)], q["gold"], args.k)
                       for q in subset]
                n = len(agg)
                rows.append({
                    "retriever": "bm25", "strategy": strategy, "query_lang": q_lang, "index_lang": index_lang,
                    "n": n, f"hit@{args.k}": round(sum(a[0] for a in agg) / n, 3),
                    f"recall@{args.k}": round(sum(a[1] for a in agg) / n, 3),
                    f"mrr@{args.k}": round(sum(a[2] for a in agg) / n, 3),
                })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    keys = list(rows[0].keys())
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(str(r[k]) for k in keys))
    failed = False
    for g in args.gate:
        strategy, lang, metric, minimum = g.split(":")
        row = next(r for r in rows if r["strategy"] == strategy and r["query_lang"] == lang
                   and r["index_lang"] == lang)
        ok = row[metric] >= float(minimum)
        print(f"gate {g}: {row[metric]} {'ok' if ok else 'FAILED'}")
        failed |= not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
