"""Bundle recorded benchmark answers for the UI's replay mode (works without a GPU or the API).

Joins eval/questions.jsonl, the graph+rerank contexts, qwen2.5:7b answers on those contexts,
and the per-question scores into app/replay.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexrag.generation.prompt import cite_label  # noqa: E402
from lexrag.serve.service import eurlex_url  # noqa: E402

RUN = "qwen2.5_7b__rag__graph"


def jl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.open(encoding="utf-8")]


def main() -> None:
    qs = {q["id"]: q for q in jl(ROOT / "eval" / "questions.jsonl")}
    ctx = {r["id"]: r["contexts"] for r in jl(ROOT / "eval" / "contexts" / "graph_rerank_top5.jsonl")}
    gens = {r["id"]: r for r in jl(ROOT / "eval" / "generations" / f"{RUN}.jsonl")}
    scores = {r["id"]: r for r in json.loads((ROOT / "eval" / "results" / "generation_per_question.json").read_text())[RUN]}
    out = []
    for qid, q in qs.items():
        g, s = gens.get(qid), scores.get(qid, {})
        if g is None:
            continue
        cited = set(g.get("sources", []))
        out.append({
            "id": qid, "question": q["question"], "lang": q["lang"], "category": q["category"],
            "answerable": q["answerable"], "reference": q.get("answer", ""),
            "answer": g["answer"], "insufficient": g.get("insufficient", False),
            "correct": s.get("correct"), "refused": s.get("refused"), "latency_s": g.get("latency_s"),
            "model": g["model"],
            "citations": [{"n": i, "label": cite_label(c), "header": c["header"], "text": c["text"],
                           "url": eurlex_url(c["provision_ids"][0], c["lang"]), "cited": i in cited, "kind": c["kind"]}
                          for i, c in enumerate(ctx.get(qid, []), 1)],
        })
    path = ROOT / "app" / "replay.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} answers to {path}")


if __name__ == "__main__":
    main()
