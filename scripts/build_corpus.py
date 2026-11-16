"""Build the corpus in two stages.

1. parse  (needs data/raw/, see download_corpus.py): EUR-Lex HTML -> provisions_{lang}.jsonl
2. chunk  (needs only the committed provisions files): provisions -> chunks_{strategy}_{lang}.jsonl

Usage:
  python scripts/build_corpus.py            # parse if data/raw exists, then chunk
  python scripts/build_corpus.py --chunk-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexrag.corpus.chunk import STRATEGIES  # noqa: E402
from lexrag.corpus.parse import Provision, parse_source  # noqa: E402
from lexrag.corpus.sources import CURRENT, LANGS, ORIGINAL_AIA  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_provisions(path: Path) -> list[Provision]:
    return [Provision.from_dict(json.loads(line)) for line in path.open(encoding="utf-8")]


def parse_stage() -> None:
    for LANG in LANGS:
        lang = LANG.lower()
        provs = []
        for src in CURRENT:
            provs += parse_source(RAW / src.filename(LANG), src.key, lang, src.role, src.version, src.celex)
        write_jsonl(OUT / f"provisions_{lang}.jsonl", (p.to_dict() for p in provs))
        orig = parse_source(RAW / ORIGINAL_AIA.filename(LANG), "AIA", lang, "provisions",
                            ORIGINAL_AIA.version, ORIGINAL_AIA.celex)
        write_jsonl(OUT / f"aia_original_provisions_{lang}.jsonl", (p.to_dict() for p in orig))
        print(f"parsed {lang}: {len(provs)} provisions, {len(orig)} in the 2024 AI Act")


def chunk_stage() -> dict:
    stats: dict = {}
    for LANG in LANGS:
        lang = LANG.lower()
        provs = read_provisions(OUT / f"provisions_{lang}.jsonl")
        s = {"provisions": dict(Counter(f"{p.doc}:{p.kind}" for p in provs)),
             "words": sum(len(p.text.split()) for p in provs),
             "aia_provisions_amended_2026": sum(1 for p in provs if p.amended_by and p.doc == "AIA")}
        for name, fn in STRATEGIES.items():
            chunks = fn(provs)
            write_jsonl(OUT / f"chunks_{name}_{lang}.jsonl", (c.to_dict() for c in chunks))
            lens = sorted(c.n_words for c in chunks)
            s[f"chunks_{name}"] = {
                "n": len(chunks), "median_words": lens[len(lens) // 2], "p95_words": lens[int(len(lens) * 0.95)],
                "max_words": lens[-1], "with_xrefs": sum(1 for c in chunks if c.xrefs),
            }
        stats[lang] = s
    (OUT / "corpus_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.chunk_only:
        if RAW.exists():
            parse_stage()
        else:
            print("data/raw missing: using committed provisions (run scripts/download_corpus.py to re-parse)")
    print(json.dumps(chunk_stage(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
