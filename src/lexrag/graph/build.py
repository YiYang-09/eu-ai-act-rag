"""Build the legal knowledge graph from parsed chunks, with rules only (no LLM).

Nodes
  provision  AIA:art_26, GDPR:art_4, AIA:anx_III, AIA:rct_50       (kind="provision")
  chunk      en:AIA:art_26:6 ...                                    (kind="chunk")
  term       term:AIA:deployer, term:GDPR:personal data             (kind="term")

Edges
  CONTAINS     provision -> chunk         from the ELI hierarchy
  REFERS_TO    chunk -> provision         cross-references extracted in parsing ("Article 6(2)", "artikel 6.2")
  DEFINED_IN   term -> chunk              Art. 3 AI Act / Art. 4 GDPR definitions
  SAME_AS      term -> term               an AI Act definition that points to a GDPR definition
  APPLIES_FROM provision -> chunk         every article -> the act's entry-into-force article
                                          (AI Act Art. 113, GDPR Art. 99), used for date questions
The graph is built per language; ids of provisions are shared, so EN and SV graphs align.
"""

from __future__ import annotations

import re
from collections import defaultdict

import networkx as nx

DEFINITION_ARTICLES = {"AIA": "AIA:art_3", "GDPR": "GDPR:art_4"}
APPLICATION_ARTICLES = {"AIA": "AIA:art_113", "GDPR": "GDPR:art_99"}

# "‘small mid-cap enterprise’ or ‘SMC’ means ..." -> both terms
_EN_DEF = re.compile(r"^\((\d+[a-z]?)\)\s+((?:‘[^’]+’(?:\s+or\s+)?)+)\s+means", re.MULTILINE)
_EN_QUOTED = re.compile(r"‘([^’]+)’")
# Swedish: "4. tillhandahållare: en fysisk ..." / "1) personuppgifter: varje upplysning ..."
_SV_DEF = re.compile(r"^(\d+[a-z]?)\)?\.?\s+([^:\n]{2,80}?):\s", re.MULTILINE)
_SAME_AS = {
    "en": re.compile(r"Article 4, points? \((\d+)\),? of Regulation \(EU\) 2016/679"),
    "sv": re.compile(r"artikel 4\.(\d+) i förordning \(EU\) 2016/679", re.IGNORECASE),
}


def extract_definitions(chunks: list[dict], lang: str) -> list[dict]:
    """Return [{doc, number, terms, chunk_id, text}] for every legal definition."""
    out = []
    for c in chunks:
        doc = c["doc"]
        if DEFINITION_ARTICLES.get(doc) not in c["provision_ids"]:
            continue
        if lang == "en":
            for m in _EN_DEF.finditer(c["text"]):
                terms = _EN_QUOTED.findall(m.group(2))
                line_end = c["text"].find("\n", m.start())
                text = c["text"][m.start(): None if line_end < 0 else line_end]
                out.append({"doc": doc, "number": m.group(1), "terms": terms, "chunk_id": c["chunk_id"], "text": text})
        else:
            for m in _SV_DEF.finditer(c["text"]):
                term = m.group(2).strip().strip("”“\"")
                if len(term.split()) > 6:
                    continue
                line_end = c["text"].find("\n", m.start())
                text = c["text"][m.start(): None if line_end < 0 else line_end]
                terms = [t.strip() for t in re.split(r"\s+eller\s+", term)]
                out.append({"doc": doc, "number": m.group(1), "terms": terms, "chunk_id": c["chunk_id"], "text": text})
    return out


def term_id(doc: str, term: str) -> str:
    return f"term:{doc}:{term.lower()}"


def build_graph(chunks: list[dict], lang: str) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph(lang=lang)
    by_prov = defaultdict(list)
    for c in chunks:
        pid = c["provision_ids"][0]
        by_prov[pid].append(c["chunk_id"])
        g.add_node(c["chunk_id"], kind="chunk", doc=c["doc"], provision=pid)
        g.add_node(pid, kind="provision", doc=c["doc"])
        g.add_edge(pid, c["chunk_id"], rel="CONTAINS")
        for ref in c["xrefs"]:
            if ":" in ref:  # skip document-level links such as "GDPR"
                g.add_node(ref, kind="provision", doc=ref.split(":")[0])
                g.add_edge(c["chunk_id"], ref, rel="REFERS_TO")

    defs = extract_definitions(chunks, lang)
    gdpr_by_number = {d["number"]: d for d in defs if d["doc"] == "GDPR"}
    for d in defs:
        for t in d["terms"]:
            tid = term_id(d["doc"], t)
            g.add_node(tid, kind="term", doc=d["doc"], label=t, number=d["number"])
            g.add_edge(tid, d["chunk_id"], rel="DEFINED_IN")
        if d["doc"] == "AIA":
            m = _SAME_AS[lang].search(d["text"])
            if m and m.group(1) in gdpr_by_number:
                target = gdpr_by_number[m.group(1)]
                for t in d["terms"]:
                    for t2 in target["terms"]:
                        g.add_edge(term_id("AIA", t), term_id("GDPR", t2), rel="SAME_AS")
                        g.add_edge(term_id("GDPR", t2), term_id("AIA", t), rel="SAME_AS")

    # the same defined term in both acts ("biometric data", "consent") is one concept
    labels = defaultdict(set)
    for d in defs:
        for t in d["terms"]:
            labels[t.lower()].add(d["doc"])
    for label, docs in labels.items():
        if docs == {"AIA", "GDPR"}:
            a, b = term_id("AIA", label), term_id("GDPR", label)
            if not g.has_edge(a, b):
                g.add_edge(a, b, rel="SAME_AS")
                g.add_edge(b, a, rel="SAME_AS")

    for doc, app in APPLICATION_ARTICLES.items():
        for pid in list(by_prov):
            if pid.startswith(f"{doc}:art_") and pid != app:
                for cid in by_prov.get(app, []):
                    g.add_edge(pid, cid, rel="APPLIES_FROM")
    g.graph["by_provision"] = dict(by_prov)
    g.graph["definitions"] = defs
    return g


def graph_stats(g: nx.MultiDiGraph) -> dict:
    kinds = defaultdict(int)
    for _, d in g.nodes(data=True):
        kinds[d.get("kind")] += 1
    rels = defaultdict(int)
    for _, _, d in g.edges(data=True):
        rels[d["rel"]] += 1
    cross = sum(1 for u, v, d in g.edges(data=True) if d["rel"] == "REFERS_TO" and g.nodes[u]["doc"] != v.split(":")[0])
    return {"nodes": dict(kinds), "edges": dict(rels), "cross_regulation_refs": cross}
