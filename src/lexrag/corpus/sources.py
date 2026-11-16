"""Registry of legal sources pulled from EUR-Lex.

Each source is one CELEX document in one language. Consolidated texts (CELEX
starting with 0) give the law as currently in force but omit recitals, so
recitals are taken from the original Official Journal (OJ) publication.
"""

from __future__ import annotations

from dataclasses import dataclass

EURLEX_HTML = "https://eur-lex.europa.eu/legal-content/{lang}/TXT/HTML/?uri=CELEX:{celex}"
LANGS = ("EN", "SV")


@dataclass(frozen=True)
class Source:
    key: str  # short document key used in ids, e.g. "AIA"
    celex: str
    role: str  # "provisions" (articles + annexes) or "recitals"
    version: str  # date the text is valid from (consolidation date or OJ date)
    title: str

    def url(self, lang: str) -> str:
        return EURLEX_HTML.format(lang=lang, celex=self.celex)

    def filename(self, lang: str) -> str:
        return f"{self.celex}_{lang}.html"


# Current law: the AI Act as amended by the Digital Omnibus on AI
# (Regulation (EU) 2026/1744), and the GDPR with its corrigenda.
CURRENT = [
    Source("AIA", "02024R1689-20260727", "provisions", "2026-07-27",
           "Artificial Intelligence Act, Regulation (EU) 2024/1689 (consolidated)"),
    Source("AIA", "32024R1689", "recitals", "2024-07-12",
           "Artificial Intelligence Act, Regulation (EU) 2024/1689 (OJ)"),
    Source("GDPR", "02016R0679-20160504", "provisions", "2016-05-04",
           "General Data Protection Regulation, Regulation (EU) 2016/679 (consolidated)"),
    Source("GDPR", "32016R0679", "recitals", "2016-05-04",
           "General Data Protection Regulation, Regulation (EU) 2016/679 (OJ)"),
]

# The AI Act as originally published, used to simulate an amendment arriving
# and to test incremental re-indexing (only changed provisions are re-embedded).
ORIGINAL_AIA = Source("AIA", "32024R1689", "provisions", "2024-07-12",
                      "Artificial Intelligence Act, Regulation (EU) 2024/1689 (OJ, as adopted)")

# The amending act itself, kept for reference and amendment questions.
OMNIBUS = Source("OMNIBUS", "32026R1744", "provisions", "2026-07-24",
                 "Digital Omnibus on AI, Regulation (EU) 2026/1744")

ALL_SOURCES = CURRENT + [OMNIBUS]

# Acts that cross-references may point to, by their "YYYY/NNN" number.
KNOWN_ACTS = {"2024/1689": "AIA", "2016/679": "GDPR", "2026/1744": "OMNIBUS"}
