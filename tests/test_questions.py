import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "data/processed/provisions_en.jsonl").exists(), reason="corpus not built")
def test_question_set_is_consistent_with_corpus():
    import subprocess
    import sys

    r = subprocess.run([sys.executable, str(ROOT / "eval/check_questions.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_question_schema():
    qs = [json.loads(line) for line in (ROOT / "eval/questions.jsonl").open(encoding="utf-8")]
    assert len(qs) >= 80
    for q in qs:
        assert q["lang"] in {"en", "sv"}
        assert q["answerable"] == bool(q["gold"]) or q["category"] == "false_premise"
