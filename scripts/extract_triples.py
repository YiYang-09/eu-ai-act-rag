"""GraphRAG round 2: let a local LLM extract (subject, relation, object) triples from each chunk.

Runs over the English article and annex chunks of both acts (recitals are skipped: they do
not create obligations). The Swedish graph reuses the English triples through the aligned
chunk ids (en:AIA:art_26:1 <-> sv:AIA:art_26:1).

  python scripts/extract_triples.py --model qwen2.5:7b            # resumable
  python scripts/extract_triples.py --limit 20                    # smoke test

Output: data/graph/triples_<model>.jsonl, one line per chunk:
  {"chunk_id", "triples": [{"s", "r", "o"}], "latency_s", "prompt_tokens", "output_tokens", "parse_ok"}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RELATIONS = ["MUST", "MUST_NOT", "MAY", "DEFINED_AS", "APPLIES_TO", "EXEMPTS", "PENALTY", "APPLIES_FROM", "REQUIRES"]

SYSTEM = f"""You build a knowledge graph of EU law. From the legal text, extract the legally meaningful
facts as triples. Rules:
- subject and object are short noun phrases (1-6 words) naming actors, systems, documents,
  obligations, data categories, authorities, dates or amounts, in singular, lowercase,
  e.g. "deployer", "high-risk ai system", "fundamental rights impact assessment".
- relation is one of: {", ".join(RELATIONS)}.
  MUST / MUST_NOT / MAY: actor -> action or obligation ("provider" MUST "technical documentation").
  DEFINED_AS: term -> short definition head. APPLIES_TO: rule or obligation -> who or what it covers.
  EXEMPTS: rule -> what is excluded. PENALTY: infringement -> fine amount. APPLIES_FROM: provision or
  obligation -> date. REQUIRES: thing -> prerequisite.
- Only facts stated in the text. At most 8 triples. If there are none, return an empty list.
Answer with JSON only: {{"triples": [{{"s": "...", "r": "...", "o": "..."}}]}}"""


def load_chunks() -> list[dict]:
    path = ROOT / "data" / "processed" / "chunks_structural_en.jsonl"
    rows = [json.loads(x) for x in path.open(encoding="utf-8")]
    return [c for c in rows if c["kind"] in ("article", "annex")]


def parse(text: str) -> tuple[list[dict], bool]:
    try:
        d = json.loads(text)
        out = []
        for t in d.get("triples", []):
            s, r, o = str(t.get("s", "")).strip().lower(), str(t.get("r", "")).strip().upper(), str(t.get("o", "")).strip().lower()
            if s and o and r in RELATIONS and len(s) <= 80 and len(o) <= 120:
                out.append({"s": s, "r": r, "o": o})
        return out[:8], True
    except (json.JSONDecodeError, AttributeError, TypeError):
        return [], False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    out = ROOT / "data" / "graph" / f"triples_{args.model.replace(':', '_').replace('/', '_')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["chunk_id"] for x in out.open(encoding="utf-8")} if out.exists() else set()
    todo = [c for c in load_chunks() if c["chunk_id"] not in done][: args.limit]
    print(f"{len(done)} done, {len(todo)} to go -> {out}")
    t_all = time.perf_counter()
    with out.open("a", encoding="utf-8") as f:
        for i, c in enumerate(todo, 1):
            body = {"model": args.model, "stream": False, "format": "json",
                    "options": {"temperature": 0, "seed": 0, "num_ctx": 4096, "num_predict": 512},
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": f"{c['header']}\n\n{c['text']}"}]}
            t0 = time.perf_counter()
            r = requests.post(f"{args.ollama}/api/chat", json=body, timeout=600).json()
            triples, ok = parse(r["message"]["content"])
            f.write(json.dumps({"chunk_id": c["chunk_id"], "triples": triples, "parse_ok": ok,
                                "latency_s": round(time.perf_counter() - t0, 2),
                                "prompt_tokens": r.get("prompt_eval_count"), "output_tokens": r.get("eval_count")},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                el = time.perf_counter() - t_all
                print(f"{i}/{len(todo)}  {el / i:.1f} s/chunk  eta {el / i * (len(todo) - i) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
