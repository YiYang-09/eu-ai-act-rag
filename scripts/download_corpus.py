"""Download the EUR-Lex HTML sources into data/raw/.

Usage: python scripts/download_corpus.py [--force]
EUR-Lex content is reusable under Commission Decision 2011/833/EU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lexrag.corpus.sources import ALL_SOURCES, LANGS  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
HEADERS = {"User-Agent": "lexrag-corpus-builder/0.1 (research; non-commercial)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    seen = set()
    for src in ALL_SOURCES:
        for lang in LANGS:
            out = RAW / src.filename(lang)
            if out in seen:
                continue
            seen.add(out)
            if out.exists() and not args.force:
                print(f"skip  {out.name} (exists)")
                continue
            r = requests.get(src.url(lang), headers=HEADERS, timeout=120)
            r.raise_for_status()
            if len(r.content) < 200_000:
                raise RuntimeError(f"{out.name}: response too small ({len(r.content)} bytes); "
                                   "EUR-Lex may have served an error page")
            out.write_bytes(r.content)
            print(f"saved {out.name} ({len(r.content) / 1e6:.1f} MB)")
            time.sleep(1.0)


if __name__ == "__main__":
    main()
