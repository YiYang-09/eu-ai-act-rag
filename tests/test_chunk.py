from lexrag.corpus.chunk import chunk_article, chunk_fixed, chunk_structural
from lexrag.corpus.parse import Provision, Unit


def defs_article(n=40):
    pts = [(f"({i})", f"term {i} means " + "word " * 20) for i in range(1, n + 1)]
    text = "Definitions apply:\n" + "\n".join(f"{a} {b}" for a, b in pts)
    return Provision(doc="AIA", lang="en", kind="article", num="3", title="Definitions",
                     units=[Unit(num=None, text=text, lead="Definitions apply:", points=pts)])


def test_long_unit_split_at_points_keeps_lead():
    chunks = chunk_structural([defs_article()], max_words=200)
    assert len(chunks) > 1
    assert all(c.text.startswith("Definitions apply:") for c in chunks)
    assert all(c.n_words <= 200 + 30 for c in chunks)
    assert chunks[0].header.endswith("Article 3 – Definitions [(1)–(8)]")


def test_article_strategy_one_chunk_per_provision():
    assert len(chunk_article([defs_article()])) == 1


def test_fixed_windows_record_overlapping_provisions():
    a = defs_article(10)
    b = Provision(doc="AIA", lang="en", kind="article", num="4", title="", units=[Unit(num=None, text="x " * 300)])
    chunks = chunk_fixed([a, b], window=200, overlap=40)
    assert any(set(c.provision_ids) == {"AIA:art_3", "AIA:art_4"} for c in chunks)
