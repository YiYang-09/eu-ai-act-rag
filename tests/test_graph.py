import numpy as np

from lexrag.graph.build import build_graph, extract_definitions, graph_stats
from lexrag.graph.expand import GraphExpander


def ch(cid, pid, text, xrefs=(), doc="AIA"):
    return {"chunk_id": cid, "provision_ids": [pid], "doc": doc, "text": text, "xrefs": list(xrefs)}


CHUNKS = [
    ch("en:AIA:art_3:0", "AIA:art_3", "For the purposes of this Regulation:\n(4) ‘deployer’ means a person using an AI system;\n"
       "(34) ‘biometric data’ means personal data resulting from specific technical processing;"),
    ch("en:GDPR:art_4:0", "GDPR:art_4", "(14) ‘biometric data’ means personal data resulting from specific processing;",
       doc="GDPR"),
    ch("en:AIA:art_6:2", "AIA:art_6", "AI systems referred to in Annex III shall be considered high-risk.", ["AIA:anx_III"]),
    ch("en:AIA:anx_III:4", "AIA:anx_III", "Employment: AI systems to analyse and filter job applications."),
    ch("en:AIA:art_113:0", "AIA:art_113", "It shall apply from 2 August 2026. Chapter III ... 2 December 2027.",
       ["AIA:art_6", "AIA:anx_III"]),
]


def test_definitions_and_same_as():
    defs = extract_definitions(CHUNKS, "en")
    assert {(d["doc"], d["number"], d["terms"][0]) for d in defs} == {
        ("AIA", "4", "deployer"), ("AIA", "34", "biometric data"), ("GDPR", "14", "biometric data")}
    g = build_graph(CHUNKS, "en")
    assert g.has_edge("term:AIA:biometric data", "term:GDPR:biometric data")
    stats = graph_stats(g)
    assert stats["edges"]["REFERS_TO"] == 3 and stats["edges"]["APPLIES_FROM"] == 2


def test_expansion_adds_definitions_references_and_dates():
    g = build_graph(CHUNKS, "en")
    vecs = np.eye(len(CHUNKS), dtype=np.float32)
    ex = GraphExpander(g, CHUNKS, vecs, max_term_df=1.0)
    q = np.ones(len(CHUNKS), dtype=np.float32)
    seeds = [CHUNKS[2]]  # Art. 6(2)
    got = {c["chunk_id"]: why for c, why in ex.expand("From when do Annex III rules apply?", q, seeds)}
    assert got["en:AIA:anx_III:4"].startswith("cited by")
    assert got["en:AIA:art_113:0"].startswith("entry into force")
    got = {c["chunk_id"] for c, _ in ex.expand("How is biometric data defined?", q, seeds)}
    assert {"en:AIA:art_3:0", "en:GDPR:art_4:0"} <= got
