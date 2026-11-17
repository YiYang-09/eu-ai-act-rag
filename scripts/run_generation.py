"""Generate answers for the benchmark with a local Ollama model.

Reads the frozen retrieval results (eval/contexts/*.jsonl), so this machine only
needs Ollama and `requests`. Resumable: questions already in the output file are
skipped, so an interrupted run can simply be restarted.

Usage:
  python scripts/run_generation.py --model qwen2.5:7b --mode rag
  python scripts/run_generation.py --model qwen2.5:7b --mode closed     # no retrieval
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexrag.generation.prompt import build_messages  # noqa: E402


def parse_reply(text: str) -> dict:
    """Parse the model's JSON; fall back to a regex if it wrapped or broke the JSON."""
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        d = json.loads(m.group() if m else text)
        sources = [int(s) for s in d.get("sources", []) if str(s).strip().lstrip("[").rstrip("]").isdigit()]
        return {"answer": str(d.get("answer", "")).strip(), "sources": sources,
                "insufficient": bool(d.get("insufficient", False)), "parse_ok": True}
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        return {"answer": text.strip(), "sources": [int(x) for x in re.findall(r"\[(\d)\]", text)],
                "insufficient": False, "parse_ok": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["rag", "closed"], default="rag")
    ap.add_argument("--contexts", default=str(ROOT / "eval" / "contexts" / "hybrid_rerank_top5.jsonl"))
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="output cap; stops runaway generations (one closed-book run looped until timeout)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or ROOT / "eval" / "generations" / f"{args.model.replace(':', '_')}__{args.mode}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["id"] for x in out.open(encoding="utf-8")} if out.exists() else set()
    qs = [json.loads(x) for x in (ROOT / "eval" / "questions.jsonl").open(encoding="utf-8")]
    ctx = {r["id"]: r["contexts"] for r in map(json.loads, Path(args.contexts).open(encoding="utf-8"))}
    todo = [q for q in qs if q["id"] not in done]
    print(f"{args.model} {args.mode}: {len(todo)} to go, {len(done)} done", flush=True)
    for i, q in enumerate(todo, 1):
        contexts = ctx[q["id"]] if args.mode == "rag" else None
        body = {"model": args.model, "messages": build_messages(q["question"], contexts), "stream": False,
                "format": "json", "options": {"temperature": 0, "seed": 0, "num_ctx": args.num_ctx,
                            "num_predict": args.max_tokens}}
        t = time.time()
        r = requests.post(f"{args.ollama}/api/chat", json=body, timeout=600)
        r.raise_for_status()
        d = r.json()
        rec = {"id": q["id"], "model": args.model, "mode": args.mode, **parse_reply(d["message"]["content"]),
               "raw": d["message"]["content"], "latency_s": round(time.time() - t, 2),
               "prompt_tokens": d.get("prompt_eval_count"), "output_tokens": d.get("eval_count"),
               "context_ids": [c["chunk_id"] for c in contexts] if contexts else []}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(todo)}] {q['id']} {rec['latency_s']}s insufficient={rec['insufficient']}", flush=True)


if __name__ == "__main__":
    main()
