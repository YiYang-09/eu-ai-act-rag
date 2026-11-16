"""Retrieval ablation: retriever x chunking x embedding model x language pair.

Metrics over answerable questions, top-k chunks:
  hit@k       at least one gold provision among the top-k chunks
  recall@k    share of gold provisions covered by the top-k chunks
  mrr@k       reciprocal rank of the first chunk touching a gold provision
  evidence@k  the passage that contains the answer (the question's evidence string) is inside a
              top-k chunk of a gold provision; stricter than hit@k and comparable across chunkings

Usage:
  python eval/retrieval_eval.py --retrievers bm25                      # no models needed (CI)
  python eval/retrieval_eval.py                                        # everything with cached vectors
  python eval/retrieval_eval.py --gate bm25:structural:en:en:hit@5:0.70
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexrag.retrieval.bm25 import BM25Index  # noqa: E402
from lexrag.retrieval.hybrid import rrf  # noqa: E402

P = ROOT / "data" / "processed"
PAIRS = [("en", "en"), ("sv", "sv"), ("sv", "en")]  # (query language, index language)
CANDIDATES = 50  # per retriever before fusion
RERANK_POOL = 30


def load_chunks(strategy: str, lang: str) -> list[dict]:
    return [json.loads(line) for line in (P / f"chunks_{strategy}_{lang}.jsonl").open(encoding="utf-8")]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).lower()


def score(results: list[dict], q: dict, index_lang: str, k: int) -> dict:
    gold = set(q["gold"])
    covered: set[str] = set()
    rr = 0.0
    evidence = q["evidence"] if index_lang == "en" else q.get("evidence_sv", [])
    ev_found = False
    for rank, c in enumerate(results[:k], 1):
        hit = gold & set(c["provision_ids"])
        if hit and rr == 0.0:
            rr = 1.0 / rank
        covered |= hit
        if hit and evidence and any(norm(e) in norm(c["header"] + " " + c["text"]) for e in evidence):
            ev_found = True
    return {"hit": float(bool(covered)), "recall": len(covered) / len(gold), "mrr": rr,
            "evidence": float(ev_found) if evidence else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--retrievers", nargs="+", default=["bm25", "dense", "hybrid", "hybrid+rerank"])
    ap.add_argument("--models", nargs="+", default=["bge-m3", "e5-small"])
    ap.add_argument("--strategies", nargs="+", default=["structural", "article", "fixed"])
    ap.add_argument("--rerank-strategies", nargs="+", default=["structural", "article", "fixed"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "retrieval.json"))
    ap.add_argument("--gate", action="append", default=[],
                    help="retriever:strategy:query_lang:index_lang:metric:min; exit 1 if below")
    args = ap.parse_args()
    k = args.k
    all_qs = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").open(encoding="utf-8")]
    qs = [q for q in all_qs if q["answerable"]]
    needs_dense = any(r != "bm25" for r in args.retrievers)

    qvecs: dict[str, dict] = {}
    if needs_dense:
        import numpy as np

        from lexrag.retrieval.dense import INDEX_DIR, DenseIndex, build_or_load

        for m in args.models:
            ids = json.loads((INDEX_DIR / m / "questions.ids.json").read_text())
            vecs = np.load(INDEX_DIR / m / "questions.npy").astype(np.float32)
            assert ids == [q["id"] for q in all_qs], "question vectors are stale; rerun scripts/embed_corpus.py"
            qvecs[m] = dict(zip(ids, vecs))
    reranker = None
    if "hybrid+rerank" in args.retrievers:
        from lexrag.retrieval.rerank import Reranker

        reranker = Reranker(device=args.device)

    rows, per_question = [], []
    for strategy in args.strategies:
        chunks = {lang: load_chunks(strategy, lang) for lang in ("en", "sv")}
        bm25 = {lang: BM25Index(chunks[lang], lang) for lang in ("en", "sv")}
        configs: list[tuple[str, str | None]] = [("bm25", None)] if "bm25" in args.retrievers else []
        for m in args.models if needs_dense else []:
            configs += [(r, m) for r in ("dense", "hybrid") if r in args.retrievers]
        if reranker and strategy in args.rerank_strategies:
            configs.append(("hybrid+rerank", args.models[0]))
        dense = {}
        if needs_dense:
            dense = {(m, lang): DenseIndex(chunks[lang], build_or_load(chunks[lang], m, strategy, lang))
                     for m in args.models for lang in ("en", "sv")}
        for retriever, model in configs:
            for q_lang, index_lang in PAIRS:
                subset = [q for q in qs if q["lang"] == q_lang]
                scores = []
                for q in subset:
                    kw = [c for c, _ in bm25[index_lang].search(q["question"], CANDIDATES)]
                    if retriever == "bm25":
                        res = kw
                    else:
                        dn = [c for c, _ in dense[(model, index_lang)].search_vec(qvecs[model][q["id"]], CANDIDATES)]
                        if retriever == "dense":
                            res = dn
                        else:
                            res = [c for c, _ in rrf([kw, dn], k=RERANK_POOL)]
                            if retriever == "hybrid+rerank":
                                res = [c for c, _ in reranker.rerank(q["question"], res, k=k, query_id=q["id"])]
                    s = score(res, q, index_lang, k)
                    scores.append(s)
                    per_question.append({"id": q["id"], "retriever": retriever, "model": model, "strategy": strategy,
                                         "index_lang": index_lang, **s,
                                         "top": [c["chunk_id"] for c in res[:k]]})
                n = len(scores)
                ev = [s["evidence"] for s in scores if s["evidence"] is not None]
                rows.append({
                    "retriever": retriever, "model": model or "-", "strategy": strategy,
                    "query_lang": q_lang, "index_lang": index_lang, "n": n,
                    f"hit@{k}": round(sum(s["hit"] for s in scores) / n, 3),
                    f"recall@{k}": round(sum(s["recall"] for s in scores) / n, 3),
                    f"mrr@{k}": round(sum(s["mrr"] for s in scores) / n, 3),
                    f"evidence@{k}": round(sum(ev) / len(ev), 3) if ev else None,
                })
                print(" | ".join(str(v) for v in rows[-1].values()), flush=True)
            if reranker:
                reranker.save()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    out.with_name(out.stem + "_per_question.json").write_text(json.dumps(per_question) + "\n", encoding="utf-8")

    failed = False
    for g in args.gate:
        retriever, strategy, ql, il, metric, minimum = g.split(":")
        row = next(r for r in rows if (r["retriever"], r["strategy"], r["query_lang"], r["index_lang"])
                   == (retriever, strategy, ql, il))
        ok = row[metric] >= float(minimum)
        print(f"gate {g}: {row[metric]} {'ok' if ok else 'FAILED'}")
        failed |= not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
