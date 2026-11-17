"""Service, store, incremental ingestion and API, with deterministic fakes (no models, no GPU)."""

import json

import numpy as np
import pytest

from lexrag.serve.fakes import DIM, HashEmbedder, fake_service
from lexrag.serve.ingest import ingest
from lexrag.serve.service import detect_lang, eurlex_url, load_chunks, parse_json_reply
from lexrag.serve.store import NumpyStore


@pytest.fixture(scope="module")
def chunks():
    return load_chunks()


@pytest.fixture(scope="module")
def svc(chunks):
    return fake_service(chunks)


def test_helpers():
    assert detect_lang("Vad är en personuppgiftsansvarig?") == "sv"
    assert detect_lang("What is a provider?") == "en"
    assert eurlex_url("AIA:art_26", "en").endswith("CELEX:02024R1689-20260727#art_26")
    assert "CELEX:32016R0679#rct_47" in eurlex_url("GDPR:rct_47", "sv")
    assert parse_json_reply('x {"answer": "a", "sources": [2], "insufficient": false} y')["sources"] == [2]
    assert parse_json_reply("not json [1]")["sources"] == [1]


def test_ask_returns_linked_citations(svc):
    a = svc.ask("What are the obligations of deployers of high-risk AI systems?")
    assert a.lang == "en" and len(a.citations) == 5
    assert a.citations[0].cited and a.citations[0].url.startswith("https://eur-lex.europa.eu/")
    assert sum(c.kind == "recital" for c in a.citations) <= 1
    assert set(a.timings_ms) == {"retrieve", "rerank_graph", "generate"}


def test_role_filter(svc):
    for role, doc in (("dpo", "GDPR"), ("ai-governance", "AIA")):
        hits, _ = svc.retrieve("What must be documented and who is responsible?", "en", role)
        assert hits and all(c["doc"] == doc for c, _, _ in hits)
    with pytest.raises(PermissionError):
        svc.retrieve("anything", "en", "intruder")


def test_numpy_store_filters(chunks):
    emb = HashEmbedder()
    cs = [c for c in chunks if c["lang"] == "en"][:200]
    store = NumpyStore(DIM)
    store.upsert(cs, emb.encode_chunks(cs))
    hits = store.search(emb.encode_queries(["data"])[0], "en", 10, ["GDPR"])
    assert all(c["doc"] == "GDPR" for c, _ in hits)
    assert store.vectors([cs[0]["chunk_id"]]).shape == (1, DIM)


def _incremental(store, chunks):
    emb = HashEmbedder()
    calls = []

    def embed(cs):
        calls.append(len(cs))
        return emb.encode_chunks(cs)

    cs = [c for c in chunks if c["lang"] == "en"][:50]
    r1 = ingest(store, cs, embed, "en")
    assert (r1.added, r1.unchanged) == (50, 0)
    edited = [dict(c) for c in cs[:-3]]
    edited[0]["text"] += " (amended)"
    extra = {**cs[0], "chunk_id": cs[0]["chunk_id"] + ":new"}
    r2 = ingest(store, edited + [extra], embed, "en")
    assert (r2.added, r2.changed, r2.unchanged, r2.deleted) == (1, 1, 46, 3)
    assert calls == [50, 2]
    assert ingest(store, edited + [extra], embed, "en").unchanged == 48


def test_incremental_ingest_numpy(chunks):
    _incremental(NumpyStore(DIM), chunks)


def test_incremental_ingest_qdrant(chunks):
    pytest.importorskip("qdrant_client")
    from lexrag.serve.store import QdrantStore

    store = QdrantStore(path=":memory:", dim=DIM)
    _incremental(store, chunks)
    q = HashEmbedder().encode_queries(["personal data"])[0]
    assert all(c["doc"] == "AIA" for c, _ in store.search(q, "en", 5, ["AIA"]))


def test_api(svc):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from lexrag.serve.api import create_app

    client = TestClient(create_app(svc))
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/ask", json={"question": "Vad är en personuppgiftsansvarig?", "role": "dpo"})
    assert r.status_code == 200
    body = r.json()
    assert body["lang"] == "sv" and all("GDPR" in c["provision_id"] for c in body["citations"])
    assert client.post("/search", json={"question": "fines", "role": "nobody"}).status_code == 403
    assert client.post("/ask", json={"question": "x"}).status_code == 422


def test_replay_bundle_matches_questions():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    rows = json.loads((root / "app" / "replay.json").read_text(encoding="utf-8"))
    n_q = sum(1 for _ in (root / "eval" / "questions.jsonl").open(encoding="utf-8"))
    assert len(rows) == n_q and all(r["citations"] for r in rows if r["answerable"])
