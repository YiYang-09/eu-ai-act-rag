"""Answer-key matching used to score generated answers automatically."""

from __future__ import annotations

import re


def _norm(s: str) -> str:
    s = s.replace(" ", " ").replace(" ", " ").replace("’", "'")
    return re.sub(r"\s+", " ", s)


def matches(text: str, groups: list[list[str]]) -> bool:
    """True if every group has at least one alternative regex matching the text."""
    if not groups:
        return False
    t = _norm(text)
    return all(any(re.search(alt, t, re.IGNORECASE) for alt in group) for group in groups)
