import importlib.util
import json
from pathlib import Path

from lexrag.generation.keys import matches
from lexrag.generation.prompt import build_messages, cite_label

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_generation", ROOT / "scripts" / "run_generation.py")
run_generation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_generation)


def test_cite_labels():
    assert cite_label({"provision_ids": ["AIA:art_26"], "unit": "6"}) == "AI Act Art. 26(6)"
    assert cite_label({"provision_ids": ["GDPR:art_3"], "unit": None}) == "GDPR Art. 3"
    assert cite_label({"provision_ids": ["AIA:anx_III"], "unit": "4"}) == "AI Act Annex III, point 4"
    assert cite_label({"provision_ids": ["AIA:rct_50"], "unit": None}) == "AI Act Recital 50"


def test_messages_number_the_sources():
    ctx = [{"provision_ids": ["AIA:art_26"], "unit": "6", "header": "h", "text": "keep logs six months"}]
    msgs = build_messages("How long?", ctx)
    assert "[1] AI Act Art. 26(6)" in msgs[1]["content"]
    assert "Sources" not in build_messages("How long?", None)[1]["content"]


def test_parse_reply_json_and_fallback():
    ok = run_generation.parse_reply('{"answer": "Six months (Article 26(6)).", "sources": [1, "2"], "insufficient": false}')
    assert ok == {"answer": "Six months (Article 26(6)).", "sources": [1, 2], "insufficient": False, "parse_ok": True}
    bad = run_generation.parse_reply("Six months [1]")
    assert bad["parse_ok"] is False and bad["sources"] == [1]


def test_keys_on_reference_answers():
    qs = [json.loads(x) for x in (ROOT / "eval" / "questions.jsonl").open(encoding="utf-8")]
    q = next(q for q in qs if q["id"] == "amd-01")
    assert matches(q["answer"], q["keys"]) and not matches(q["answer_2024"], q["keys"])
