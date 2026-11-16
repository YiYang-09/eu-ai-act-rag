"""Extract cross-references ("Article 6(2)", "Annex III", "artikel 6.2", "bilaga III").

References are resolved to provision ids such as ``AIA:art_6`` or
``GDPR:art_9``. A reference followed by "of Regulation (EU) 2016/679" (or the
Swedish "i förordning (EU) 2016/679") points to that act; references to acts
outside the corpus are dropped. Mentions of a known act without an article
("Regulation (EU) 2016/679") become a document-level link such as ``GDPR``.
"""

from __future__ import annotations

import re

from .sources import KNOWN_ACTS

ROMAN = r"[IVXLC]+"
NUM = r"\d+[a-z]?"

# A list of article numbers with optional paragraph suffixes: "9 to 15", "8 and 9", "6(2)", "6.2"
_EN_ART_LIST = rf"{NUM}(?:\(\d+[a-z]?\))*(?:\s*(?:,|and|or|to|-|–)\s*{NUM}(?:\(\d+[a-z]?\))*)*"
_SV_ART_LIST = rf"{NUM}(?:\.\d+[a-z]?)*(?:\s*(?:,|och|eller|till|-|–)\s*{NUM}(?:\.\d+[a-z]?)*)*"

PATTERNS = {
    "en": {
        "article": re.compile(rf"\bArticles?\s+({_EN_ART_LIST})"),
        "annex": re.compile(rf"\bAnnex(?:es)?\s+({ROMAN}(?:\s*(?:,|and|or|to)\s*{ROMAN})*)\b"),
        "act": re.compile(r"\b(?:Regulation|Directive)\s+\((?:EU|EC|EEC)\)\s+(?:No\s+)?(\d{4}/\d+|\d+/\d{4})"),
        "of_act": re.compile(r"^(?:(?!Articles?\b|Annex|\band\b|\bor\b)[^.;]){0,40}?\bof\s+(?:Regulation|Directive|Decision)\s+(?:\((?:EU|EC|EEC)\)\s+)?(?:No\s+)?(\d{4}/\d+|\d+/\d{4})?"),
        "this_act": re.compile(r"^(?:(?!Articles?\b|Annex|\band\b|\bor\b)[^.;]){0,40}?\bof\s+this\s+Regulation"),
    },
    "sv": {
        "article": re.compile(rf"\bartik(?:el|eln|lar|larna)\s+({_SV_ART_LIST})", re.IGNORECASE),
        "annex": re.compile(rf"\bbilag(?:a|an|orna|or)\s+({ROMAN}(?:\s*(?:,|och|eller|till)\s*{ROMAN})*)\b", re.IGNORECASE),
        "act": re.compile(r"\b(?:förordning|direktiv)\s+\((?:EU|EG|EEG)\)\s+(?:nr\s+)?(\d{4}/\d+|\d+/\d{4})", re.IGNORECASE),
        "of_act": re.compile(r"^(?:(?!artik|bilag|\boch\b|\beller\b)[^.;]){0,40}?\bi\s+(?:Europaparlamentets\s+och\s+rådets\s+|rådets\s+|kommissionens\s+)?(?:förordning|direktiv|beslut)\s+(?:\((?:EU|EG|EEG)\)\s+)?(?:nr\s+)?(\d{4}/\d+|\d+/\d{4})?", re.IGNORECASE),
        "this_act": re.compile(r"^(?:(?!artik|bilag|\boch\b|\beller\b)[^.;]){0,40}?\bi\s+denna\s+förordning", re.IGNORECASE),
    },
}

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def roman_to_int(s: str) -> int:
    total, prev = 0, 0
    for ch in reversed(s):
        v = _ROMAN_VALUES[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


def int_to_roman(n: int) -> str:
    out = []
    for v, sym in [(100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def _expand(listing: str, lang: str, roman: bool = False) -> list[str]:
    """'9 to 15' -> 9..15, '8 and 9' -> 8, 9, '6(2)' -> 6, '6.2' -> 6."""
    sep = r"\s*(,|and|or|to|och|eller|till|-|–)\s*"
    tokens = re.split(sep, listing)
    items, ops = tokens[0::2], tokens[1::2]
    base = []
    for it in items:
        it = it.strip()
        if not roman:
            it = re.split(r"[(.]", it)[0]
        base.append(it)
    out = [base[0]] if base and base[0] else []
    for op, nxt in zip(ops, base[1:]):
        if not nxt:
            continue
        if op in {"to", "till", "-", "–"} and out:
            lo, hi = out[-1], nxt
            try:
                lo_i = roman_to_int(lo) if roman else int(re.match(r"\d+", lo).group())
                hi_i = roman_to_int(hi) if roman else int(re.match(r"\d+", hi).group())
            except (KeyError, AttributeError):
                out.append(nxt)
                continue
            if 0 < hi_i - lo_i <= 60:
                out.extend(int_to_roman(i) if roman else str(i) for i in range(lo_i + 1, hi_i + 1))
                continue
        out.append(nxt)
    return out


def _target_doc(text_after: str, lang: str, self_doc: str) -> str | None:
    pats = PATTERNS[lang]
    if pats["this_act"].match(text_after):
        return self_doc
    m = pats["of_act"].match(text_after)
    if m:
        return KNOWN_ACTS.get(m.group(1) or "", None)  # None => outside the corpus
    return self_doc


def extract_xrefs(text: str, lang: str, self_doc: str, self_id: str | None = None) -> list[str]:
    """Return sorted unique provision ids (and doc-level ids) referenced in text."""
    pats = PATTERNS[lang]
    refs: list[str] = []
    for m in pats["article"].finditer(text):
        doc = _target_doc(text[m.end(): m.end() + 120], lang, self_doc)
        if doc is None:
            continue
        refs.extend(f"{doc}:art_{n}" for n in _expand(m.group(1), lang))
    for m in pats["annex"].finditer(text):
        doc = _target_doc(text[m.end(): m.end() + 120], lang, self_doc)
        if doc is None:
            continue
        refs.extend(f"{doc}:anx_{n}" for n in _expand(m.group(1), lang, roman=True))
    for m in pats["act"].finditer(text):
        doc = KNOWN_ACTS.get(m.group(1))
        if doc and doc != self_doc:
            refs.append(doc)
    out = sorted(set(refs) - {self_id})
    return out
