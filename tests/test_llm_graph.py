import json

import numpy as np

from lexrag.graph.llm_graph import CombinedExpander, LLMGraph, LLMGraphExpander, load_triples, norm_entity


def _chunks(lang):
    ids = ["AIA:art_26:1", "AIA:art_113:0", "AIA:art_27:1", "GDPR:art_83:5"]
    return [{"chunk_id": f"{lang}:{i}", "lang": lang, "doc": i.split(":")[0], "kind": "article",
             "provision_ids": [":".join(i.split(":")[:2])], "header": i, "text": i} for i in ids]


def test_norm_entity():
    assert norm_entity("The Deployers") == "deployer"
    assert norm_entity("‘high-risk AI system’") == "high-risk ai system"
    assert norm_entity("access") == "access"


def test_expand_links_by_embedding_and_projects_to_sv(tmp_path):
    rows = [
        {"chunk_id": "en:AIA:art_26:1", "triples": [{"s": "deployers", "r": "MUST", "o": "human oversight"}]},
        {"chunk_id": "en:AIA:art_113:0", "triples": [{"s": "annex III obligations", "r": "APPLIES_FROM", "o": "2 december 2027"}]},
        {"chunk_id": "en:AIA:art_27:1", "triples": [{"s": "deployer", "r": "MUST", "o": "fundamental rights impact assessment"}]},
        {"chunk_id": "en:GDPR:art_83:5", "triples": []},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    g = LLMGraph(load_triples(p))
    assert g.stats()["triples"] == 3 and "deployer" in g.entities
    ent = np.eye(len(g.entities), 4, dtype=np.float32)
    # question vector close to "annex iii obligation"
    q = ent[g.entities.index("annex iii obligation")]
    for lang in ("en", "sv"):
        cs = _chunks(lang)
        vecs = np.ones((len(cs), 4), dtype=np.float32) / 2
        ex = LLMGraphExpander(g, cs, vecs, ent, link_min=0.5)
        got = ex.expand("q", q, seeds=[cs[0]])
        ids = [c["chunk_id"] for c, _ in got]
        assert f"{lang}:AIA:art_113:0" in ids and f"{lang}:AIA:art_26:1" not in ids  # seeds excluded
        # seed shares "deployer" with art 27
        assert f"{lang}:AIA:art_27:1" in ids
        both = CombinedExpander(ex, ex).expand("q", q, [cs[0]])
        assert len(both) == len(got)
