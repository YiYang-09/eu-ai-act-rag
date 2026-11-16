"""Parse EUR-Lex HTML into structured provisions.

Handles the two HTML layouts EUR-Lex serves:

* OJ layout (original Official Journal publication): classes ``oj-*``,
  numbered paragraphs as ``<div id="006.001">``, lists as ``<table>``.
* Consolidated layout (CONSLEG): classes ``norm``/``grid-list``, numbered
  paragraphs as ``<div class="norm"><span class="no-parag">1.</span>``,
  and ``▼M1`` markers in front of text changed by an amending act.

Both layouts share ELI ids: ``art_6``, ``cpt_III``, ``cpt_III.sct_1``,
``anx_III``, ``rct_12``. These ids are identical across languages, which is
what lets an English and a Swedish chunk be aligned to the same provision.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

try:  # bs4 warns because EUR-Lex declares an XML prolog; HTML parsing is intended.
    from bs4 import XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:  # pragma: no cover
    pass

ART_ID = re.compile(r"^art_(\d+[a-z]?)$")
OJ_PARA_ID = re.compile(r"^\d{3}\.\d{3}$")
PARA_NUM = re.compile(r"^\s*(\d+[a-z]?)\.\s+")
SKIP_CLASSES = {"modref", "arrow", "footnote", "oj-note", "separator-annex", "oj-separator"}


@dataclass
class Unit:
    """A paragraph of an article, a top-level point of an annex, or a recital."""

    num: str | None  # "1", "1a", "(3)", "2." ... None for unnumbered text
    text: str
    points: list[tuple[str, str]] = field(default_factory=list)  # top-level (label, text)
    lead: str = ""  # text before the first top-level point
    amended_by: list[str] = field(default_factory=list)  # CELEX numbers of amending acts


@dataclass
class Provision:
    doc: str  # "AIA" | "GDPR" | "OMNIBUS"
    lang: str  # "en" | "sv"
    kind: str  # "article" | "annex" | "recital"
    num: str  # "6", "4a", "III", "12"
    title: str
    chapter: str = ""  # "CHAPTER III — HIGH-RISK AI SYSTEMS"
    section: str = ""
    units: list[Unit] = field(default_factory=list)
    amended_by: list[str] = field(default_factory=list)
    version: str = ""
    celex: str = ""

    @property
    def pid(self) -> str:
        prefix = {"article": "art", "annex": "anx", "recital": "rct"}[self.kind]
        return f"{self.doc}:{prefix}_{self.num}"

    @property
    def text(self) -> str:
        return "\n".join(u.text for u in self.units)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["id"] = self.pid
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Provision":
        d = {k: v for k, v in d.items() if k != "id"}
        d["units"] = [Unit(**{**u, "points": [tuple(p) for p in u["points"]]}) for u in d["units"]]
        return cls(**d)


# ----------------------------------------------------------------------------
# text rendering


def _classes(el: Tag) -> set[str]:
    return set(el.get("class") or [])


def _clean(s: str) -> str:
    s = s.replace("\xa0", " ").replace(" ", " ").replace(" ", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def _is_note_ref(el: Tag) -> bool:
    href = el.get("href") or ""
    return el.name == "a" and (
        href.startswith("#E") or href.startswith("#ntr") or href.startswith("#src") or "oj-note-tag" in _classes(el)
    )


def _list_item(el: Tag) -> tuple[Tag, Tag] | None:
    """Return (label cell, body cell) if el is a one-row list item."""
    cls = _classes(el)
    if el.name == "div" and "grid-list" in cls:
        c1 = el.find("div", class_="grid-list-column-1", recursive=False)
        c2 = el.find("div", class_="grid-list-column-2", recursive=False)
        if c1 is not None and c2 is not None:
            return c1, c2
    if el.name == "table":
        body = el.find("tbody", recursive=False) or el
        rows = body.find_all("tr", recursive=False)
        if len(rows) == 1:
            tds = rows[0].find_all("td", recursive=False)
            if len(tds) == 2:
                return tds[0], tds[1]
    return None


def render(el) -> str:
    """Render an element to plain text, keeping list labels next to their text."""
    if isinstance(el, NavigableString):
        return str(el)
    if not isinstance(el, Tag):
        return ""
    cls = _classes(el)
    if cls & SKIP_CLASSES or el.name in {"script", "style", "col", "hr"}:
        return ""
    if _is_note_ref(el):
        return ""
    if "superscript" in cls or "oj-super" in cls:
        return "^" + el.get_text(strip=True)
    item = _list_item(el)
    if item is not None:
        label = _clean(render(item[0]))
        body = _clean(render(item[1]))
        return f"\n{label} {body}\n"
    if el.name == "table":  # real data table: one line per row, cells separated by " | "
        rows = []
        for tr in el.find_all("tr"):
            cells = [_clean(render(td)) for td in tr.find_all(["td", "th"], recursive=False)]
            rows.append(" | ".join(c for c in cells if c))
        return "\n" + "\n".join(r for r in rows if r) + "\n"
    inner = "".join(render(c) for c in el.children)
    if el.name in {"p", "div", "br", "tr", "li"}:
        return "\n" + inner + "\n"
    return inner


# ----------------------------------------------------------------------------
# structure


def _modref_acts(el: Tag) -> list[str] | None:
    """For a ▼ marker return the amending acts it introduces ([] for ▼B), else None."""
    if not (isinstance(el, Tag) and "modref" in _classes(el)):
        return None
    acts = []
    for a in el.find_all("a"):
        title = a.get("title") or ""
        celex = title.split(":")[0].strip()
        if a.get_text(strip=True).startswith("▼B"):
            continue
        if celex:
            acts.append(celex)
    return acts


def _para_start(el: Tag) -> str | None:
    """Paragraph number if this child element opens a numbered paragraph."""
    if not isinstance(el, Tag):
        return None
    if el.name == "div" and OJ_PARA_ID.match(el.get("id") or ""):
        m = PARA_NUM.match(_clean(render(el)))
        return m.group(1) if m else el["id"].split(".")[1].lstrip("0")
    if el.name == "div" and "norm" in _classes(el):
        span = el.find("span", class_="no-parag", recursive=False)
        if span is not None:
            return span.get_text(strip=True).rstrip(".").strip()
    return None


def _split_points(el: Tag) -> tuple[str, list[tuple[str, str]]]:
    """Split an element into lead text and its top-level list items."""
    lead_parts, points = [], []
    container = el
    # consolidated layout nests the paragraph body one level down
    inner = el.find("div", class_="inline-element", recursive=False) if isinstance(el, Tag) else None
    if inner is not None:
        container = inner
    for child in container.children:
        item = _list_item(child) if isinstance(child, Tag) else None
        if item is not None:
            points.append((_clean(render(item[0])), _clean(render(item[1]))))
        elif not points:
            lead_parts.append(render(child))
        else:  # text after the list (closing subparagraph) goes to the last point
            extra = _clean(render(child))
            if extra:
                lab, txt = points[-1]
                points[-1] = (lab, txt + "\n" + extra)
    return _clean("".join(lead_parts)), points


def _units_from_children(children: list, numbered: bool) -> list[Unit]:
    """Group the body elements of an article/annex into units.

    numbered=True: articles, units open at numbered paragraphs.
    numbered=False: annexes, units open at top-level list items.
    """
    units: list[Unit] = []
    current_acts: list[str] = []

    def new_unit(num):
        u = Unit(num=num, text="", amended_by=list(dict.fromkeys(current_acts)))
        units.append(u)
        return u

    cur: Unit | None = None
    cur_els: list[Tag] = []

    def flush():
        if cur is None:
            return
        text = _clean("".join(render(e) for e in cur_els))
        cur.text = re.sub(r"^(\(?\d+[a-z]?[.)]?)\n", r"\1 ", text)
        if len(cur_els) == 1:
            cur.lead, cur.points = _split_points(cur_els[0])
        else:
            lead, pts = [], []
            for e in cur_els:
                item = _list_item(e)
                if item is not None:
                    pts.append((_clean(render(item[0])), _clean(render(item[1]))))
                elif not pts:
                    lead.append(render(e))
                else:
                    extra = _clean(render(e))
                    if extra:
                        pts[-1] = (pts[-1][0], pts[-1][1] + "\n" + extra)
            cur.lead, cur.points = _clean("".join(lead)), pts

    for child in children:
        if isinstance(child, NavigableString):
            if not child.strip():
                continue
        acts = _modref_acts(child)
        if acts is not None:
            current_acts = acts
            continue
        if isinstance(child, Tag) and "arrow" in _classes(child):
            continue
        if numbered:
            num = _para_start(child)
            opens = num is not None
        else:
            item = _list_item(child) if isinstance(child, Tag) else None
            num = _clean(render(item[0])).rstrip(".") if item is not None else None
            opens = item is not None
            if isinstance(child, Tag) and GROUP_HEADING_CLASSES & _classes(child):
                m = re.match(r"^([0-9A-Z]+)\.", _clean(child.get_text(" ")))
                num, opens = (m.group(1) if m else None), True
        if opens or cur is None:
            flush()
            cur = new_unit(num)
            cur_els = []
        elif current_acts:
            cur.amended_by = list(dict.fromkeys(cur.amended_by + current_acts))
        cur_els.append(child)
    flush()
    return [u for u in units if u.text]


def _heading(div: Tag | None) -> str:
    if div is None:
        return ""
    ps = [p for p in div.find_all("p", recursive=False)]
    title_div = div.find("div", class_="eli-title", recursive=False)
    parts = [_clean(p.get_text(" ")) for p in ps[:1]]
    if title_div is not None:
        parts.append(_clean(title_div.get_text(" ")))
    elif len(ps) > 1:
        parts.append(_clean(ps[1].get_text(" ")))
    return " — ".join(p for p in parts if p)


def _ancestor_with_id(el: Tag, pattern: str) -> Tag | None:
    rx = re.compile(pattern)
    for parent in el.parents:
        if isinstance(parent, Tag) and rx.match(parent.get("id") or ""):
            return parent
    return None


def parse_articles(soup: BeautifulSoup, doc: str, lang: str) -> list[Provision]:
    out = []
    for art in soup.find_all("div", id=ART_ID):
        num = ART_ID.match(art["id"]).group(1)
        title_div = art.find("div", class_="eli-title", recursive=False)
        title = _clean(title_div.get_text(" ")) if title_div else ""
        body = []
        started = False
        for child in art.children:
            if child is title_div:
                started = True
                continue
            if started:
                body.append(child)
        if not started:  # no subtitle: skip the "Article N" heading paragraph
            body = [c for c in art.children if not (isinstance(c, Tag) and c.name == "p" and
                    ({"oj-ti-art", "title-article-norm"} & _classes(c)))]
        units = _units_from_children(body, numbered=True)
        chapter = _ancestor_with_id(art, r"^cpt_[IVXLC]+$")
        section = _ancestor_with_id(art, r"^cpt_[IVXLC]+\.sct_\d+$")
        prov = Provision(doc=doc, lang=lang, kind="article", num=num, title=title,
                         chapter=_heading(chapter), section=_heading(section), units=units)
        prov.amended_by = sorted({a for u in units for a in u.amended_by})
        out.append(prov)
    return out


GROUP_HEADING_CLASSES = {"title-gr-seq-level-1", "oj-ti-grseq-1"}
ANNEX_TITLE_CLASSES = {"oj-doc-ti", "title-annex-1", "title-annex-2"}


def parse_annexes(soup: BeautifulSoup, doc: str, lang: str) -> list[Provision]:
    out = []
    for anx in soup.find_all("div", id=re.compile(r"^anx_[IVXLC]+$")):
        num = anx["id"].split("_", 1)[1]
        titles = [p for p in anx.find_all("p", recursive=False) if ANNEX_TITLE_CLASSES & _classes(p)]
        if len(titles) < 2:  # some annexes put their title in an ordinary paragraph
            nxt = titles[0].find_next_sibling("p") if titles else None
            if nxt is not None and len(nxt.get_text(strip=True)) < 300:
                titles.append(nxt)
        title = _clean(titles[1].get_text(" ")) if len(titles) > 1 else ""
        body = [c for c in anx.children if not (isinstance(c, Tag) and c in titles)]
        # In the consolidated layout the modref marker sits above the annex title
        acts_before = []
        for c in anx.children:
            if isinstance(c, Tag) and c in titles:
                break
            a = _modref_acts(c) if isinstance(c, Tag) else None
            if a:
                acts_before = a
        units = _units_from_children(body, numbered=False)
        if acts_before:
            for u in units:
                u.amended_by = list(dict.fromkeys(acts_before + u.amended_by))
        prov = Provision(doc=doc, lang=lang, kind="annex", num=num, title=title, units=units)
        prov.amended_by = sorted({a for u in units for a in u.amended_by})
        out.append(prov)
    return out


def parse_recitals(soup: BeautifulSoup, doc: str, lang: str) -> list[Provision]:
    out = []
    for rct in soup.find_all("div", id=re.compile(r"^rct_\d+$")):
        num = rct["id"].split("_", 1)[1]
        text = _clean(render(rct))
        text = re.sub(r"^\(\d+\)\s*", "", text)
        out.append(Provision(doc=doc, lang=lang, kind="recital", num=num, title="",
                             units=[Unit(num=None, text=text)]))
    return out


def load_soup(path: Path) -> BeautifulSoup:
    return BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")


def parse_source(path: Path, doc: str, lang: str, role: str, version: str = "", celex: str = "") -> list[Provision]:
    soup = load_soup(path)
    if role == "recitals":
        provs = parse_recitals(soup, doc, lang)
    else:
        provs = parse_articles(soup, doc, lang) + parse_annexes(soup, doc, lang)
    for p in provs:
        p.version, p.celex = version, celex
    return provs
