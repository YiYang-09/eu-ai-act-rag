"""Prompts for grounded answering with numbered sources, and for the closed-book baseline."""

from __future__ import annotations

import re

DOC_SHORT = {"AIA": "AI Act", "GDPR": "GDPR"}
KIND_LABEL = {"art": "Art.", "anx": "Annex", "rct": "Recital"}


def cite_label(chunk: dict) -> str:
    """Human citation for a chunk: 'AI Act Art. 26(6)', 'GDPR Art. 33(1)', 'AI Act Annex III'."""
    pid = chunk["provision_ids"][0]
    doc, _, rest = pid.partition(":")
    kind, _, num = rest.partition("_")
    label = f"{DOC_SHORT.get(doc, doc)} {KIND_LABEL[kind]} {num}"
    unit = chunk.get("unit")
    if kind == "art" and unit and re.fullmatch(r"\d+[a-z]?", unit):
        label += f"({unit})"
    elif kind == "anx" and unit:
        label += f", point {unit}"
    return label


SYSTEM_RAG = """You are a compliance assistant for EU law. You answer questions about the EU AI Act \
(Regulation (EU) 2024/1689, as amended by Regulation (EU) 2026/1744) and the GDPR (Regulation (EU) 2016/679).

Rules:
1. Use ONLY the numbered sources given with the question. Do not use outside knowledge, even if you think \
the sources are incomplete or out of date.
2. Every factual statement must be supported by a source. List the numbers of the sources you used.
3. If the sources do not contain the answer, set "insufficient" to true and say briefly that the AI Act \
and GDPR texts provided do not answer the question. Do not guess.
4. If the question assumes something the sources contradict, say so and correct it.
5. Answer in the same language as the question, in 1 to 4 sentences, and name the article in the answer \
(for example "Article 26(6)").

Reply with JSON only: {"answer": "...", "sources": [numbers], "insufficient": true or false}"""

SYSTEM_CLOSED = """You are a compliance assistant for EU law. You answer questions about the EU AI Act \
(Regulation (EU) 2024/1689) and the GDPR (Regulation (EU) 2016/679) from your own knowledge.

Rules:
1. Answer in the same language as the question, in 1 to 4 sentences, and name the relevant article.
2. If the AI Act and GDPR do not answer the question, or you do not know, set "insufficient" to true and \
say so. Do not guess.
3. If the question assumes something false, say so and correct it.

Reply with JSON only: {"answer": "...", "sources": [], "insufficient": true or false}"""


def format_sources(contexts: list[dict]) -> str:
    parts = []
    for i, c in enumerate(contexts, 1):
        parts.append(f"[{i}] {cite_label(c)}\n{c['header']}\n{c['text']}")
    return "\n\n".join(parts)


def build_messages(question: str, contexts: list[dict] | None) -> list[dict]:
    if contexts is None:
        return [{"role": "system", "content": SYSTEM_CLOSED},
                {"role": "user", "content": f"Question: {question}"}]
    return [{"role": "system", "content": SYSTEM_RAG},
            {"role": "user", "content": f"Sources:\n\n{format_sources(contexts)}\n\nQuestion: {question}"}]
