"""Verify the question set against the corpus.

For every question: gold ids exist, every evidence string occurs in the
(English) text of at least one gold provision, Swedish evidence occurs in the
Swedish text, gold_units point at existing paragraphs, and every "changed"
question's evidence is absent from the 2024 text of the AI Act.
Exit code 1 on any failure.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"


def load(name):
    return {d["id"]: d for d in map(json.loads, (P / name).open(encoding="utf-8"))}


def full_text(prov: dict) -> str:
    return prov["title"] + "\n" + "\n".join(u["text"] for u in prov["units"])


def main() -> int:
    en, sv = load("provisions_en.jsonl"), load("provisions_sv.jsonl")
    old = load("aia_original_provisions_en.jsonl")
    qs = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").open(encoding="utf-8")]
    errors = []
    for q in qs:
        for g in q["gold"]:
            if g not in en:
                errors.append(f"{q['id']}: unknown gold {g}")
        texts_en = " ".join(full_text(en[g]) for g in q["gold"] if g in en)
        for ev in q["evidence"]:
            if ev not in texts_en:
                errors.append(f"{q['id']}: evidence not found: {ev!r}")
        texts_sv = " ".join(full_text(sv[g]) for g in q["gold"] if g in sv)
        for ev in q.get("evidence_sv", []):
            if ev.lower() not in texts_sv.lower():
                errors.append(f"{q['id']}: Swedish evidence not found: {ev!r}")
        for gu in q["gold_units"]:
            pid, _, num = gu.partition("#")
            if pid not in en or num not in {u["num"] for u in en[pid]["units"]}:
                errors.append(f"{q['id']}: unknown unit {gu}")
        if q["version"] == "changed":
            old_text = " ".join(full_text(old[g]) for g in q["gold"] if g in old)
            for ev in q["evidence"]:
                if ev in old_text:
                    errors.append(f"{q['id']}: 'changed' evidence also in 2024 text: {ev!r}")
            if not q.get("answer_2024"):
                errors.append(f"{q['id']}: changed question without answer_2024")
        if q["answerable"] and not q["gold"]:
            errors.append(f"{q['id']}: answerable but no gold")
    print(f"{len(qs)} questions")
    print("by category:", dict(Counter(q["category"] for q in qs)))
    print("by lang:", dict(Counter(q["lang"] for q in qs)))
    print("answerable:", sum(q["answerable"] for q in qs), "| changed by 2026 amendment:",
          sum(q["version"] == "changed" for q in qs))
    for e in errors:
        print("ERROR", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
