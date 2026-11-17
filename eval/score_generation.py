"""Score generated answers (eval/generations/*.jsonl) against the benchmark.

Per run:
  correct          answerable questions whose answer satisfies all answer keys and is not a refusal
  outdated         amendment questions answered with the pre-2026 rule (keys_2024 match)
  refusal_acc      unanswerable questions correctly refused
  false_refusal    answerable questions wrongly refused
  premise_fixed    false-premise questions where the answer corrects the premise
  cite_support     RAG answers (answered, not refused) citing at least one gold provision
  cite_precision   share of cited sources that belong to a gold provision
  unsupported      RAG answers that cite no source at all
  latency_s, output_tokens  medians

Usage: python eval/score_generation.py [files...]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexrag.generation.keys import matches  # noqa: E402

REFUSAL_HINTS = r"(do not (contain|answer|provide|specify|mention)|does not (contain|answer|provide|specify|mention)|not (contained|covered|addressed|specified) in|cannot be answered|no information|insufficient|framgår inte|innehåller inte|saknas)"


def is_refusal(rec: dict) -> bool:
    import re

    return rec["insufficient"] or bool(re.search(REFUSAL_HINTS, rec["answer"], re.IGNORECASE))


def score_run(path: Path, qs: dict, ctx: dict) -> tuple[dict, list[dict]]:
    recs = [json.loads(x) for x in path.open(encoding="utf-8")]
    rows = []
    for r in recs:
        q = qs[r["id"]]
        refused = is_refusal(r)
        row = {"id": q["id"], "category": q["category"], "lang": q["lang"], "version": q["version"],
               "refused": refused, "answer": r["answer"]}
        if q["category"] == "unanswerable":
            row["refusal_ok"] = refused
        elif q["category"] == "false_premise":
            row["premise_fixed"] = matches(r["answer"], q["keys"])
        else:
            row["correct"] = (not refused) and matches(r["answer"], q["keys"])
            if q.get("keys_2024"):
                row["outdated"] = matches(r["answer"], q["keys_2024"]) and not row["correct"]
        if r["mode"] == "rag" and not refused and q["gold"]:
            contexts = ctx[q["id"]]
            cited = [contexts[i - 1] for i in r["sources"] if 1 <= i <= len(contexts)]
            gold = set(q["gold"])
            hits = [c for c in cited if gold & set(c["provision_ids"])]
            row["cited"] = len(cited)
            row["cite_support"] = bool(hits)
            row["cite_precision"] = len(hits) / len(cited) if cited else 0.0
        row["latency_s"], row["output_tokens"] = r["latency_s"], r.get("output_tokens")
        rows.append(row)

    def mean(key, cond=lambda x: True):
        vals = [float(x[key]) for x in rows if key in x and cond(x)]
        return (round(sum(vals) / len(vals), 3), len(vals)) if vals else (None, 0)

    answerable = [x for x in rows if "correct" in x]
    summary = {
        "run": path.stem, "n": len(rows),
        "correct": mean("correct"),
        "correct_en": mean("correct", lambda x: x["lang"] == "en"),
        "correct_sv": mean("correct", lambda x: x["lang"] == "sv"),
        "correct_amendment": mean("correct", lambda x: x["version"] == "changed"),
        "outdated_amendment": mean("outdated"),
        "refusal_acc": mean("refusal_ok"),
        "false_refusal": (round(sum(x["refused"] for x in answerable) / len(answerable), 3), len(answerable)),
        "premise_fixed": mean("premise_fixed"),
        "cite_support": mean("cite_support"),
        "cite_precision": mean("cite_precision"),
        "unsupported": (round(sum(1 for x in rows if x.get("cited") == 0) / max(1, sum(1 for x in rows if "cited" in x)), 3),
         sum(1 for x in rows if "cited" in x)),
        "latency_s_median": statistics.median(x["latency_s"] for x in rows),
        "output_tokens_median": statistics.median(x["output_tokens"] or 0 for x in rows),
    }
    return summary, rows


def main() -> None:
    files = [Path(f) for f in sys.argv[1:]] or sorted((ROOT / "eval" / "generations").glob("*.jsonl"))
    qs = {q["id"]: q for q in map(json.loads, (ROOT / "eval" / "questions.jsonl").open(encoding="utf-8"))}
    ctx = {r["id"]: r["contexts"] for r in
           map(json.loads, (ROOT / "eval" / "contexts" / "hybrid_rerank_top5.jsonl").open(encoding="utf-8"))}
    summaries, details = [], {}
    for f in files:
        s, rows = score_run(f, qs, ctx)
        summaries.append(s)
        details[f.stem] = rows
        print(json.dumps(s))
    out = ROOT / "eval" / "results"
    out.mkdir(exist_ok=True)
    (out / "generation.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    (out / "generation_per_question.json").write_text(json.dumps(details, ensure_ascii=False, indent=1) + "\n",
                                                      encoding="utf-8")


if __name__ == "__main__":
    main()
