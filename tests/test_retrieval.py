from lexrag.retrieval.bm25 import BM25Index, tokenize
from lexrag.retrieval.hybrid import rrf


def chunk(cid, text):
    return {"chunk_id": cid, "header": "", "text": text, "strategy": "structural", "provision_ids": [cid]}


def test_tokenize_drops_stopwords_and_keeps_swedish_letters():
    assert tokenize("Hur länge ska loggar sparas?", "sv") == ["länge", "loggar", "sparas"]


def test_bm25_ranks_matching_chunk_first():
    chunks = [chunk("a", "logs shall be kept for six months"), chunk("b", "fines up to EUR 35 000 000"),
              chunk("c", "the board shall advise the commission")]
    assert BM25Index(chunks, "en").search("how long to keep logs", k=1)[0][0]["chunk_id"] == "a"


def test_rrf_prefers_items_ranked_by_both():
    a, b, c = chunk("a", ""), chunk("b", ""), chunk("c", "")
    fused = rrf([[a, b, c], [b, c, a]], k=3)
    assert [x["chunk_id"] for x, _ in fused][0] == "b"
