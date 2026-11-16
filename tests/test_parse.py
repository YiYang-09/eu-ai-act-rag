from bs4 import BeautifulSoup

from lexrag.corpus.parse import parse_annexes, parse_articles, parse_recitals

OJ_HTML = """
<div id="cpt_III"><p class="oj-ti-section-1">CHAPTER III</p>
<div class="eli-title" id="cpt_III.tit_1"><p class="oj-ti-section-2">HIGH-RISK AI SYSTEMS</p></div>
<div class="eli-subdivision" id="art_6">
<p class="oj-ti-art">Article 6</p>
<div class="eli-title" id="art_6.tit_1"><p class="oj-sti-art">Classification rules</p></div>
<div id="006.001"><p class="oj-normal">1.   Both conditions apply:</p>
<table><col/><col/><tbody><tr><td><p class="oj-normal">(a)</p></td><td><p class="oj-normal">safety component;</p></td></tr></tbody></table>
<table><col/><col/><tbody><tr><td><p class="oj-normal">(b)</p></td><td><p class="oj-normal">third-party assessment.</p></td></tr></tbody></table>
</div>
<div id="006.002"><p class="oj-normal">2.   Systems in Annex III are high-risk<a class="oj-note-tag" href="#ntr1">(1)</a>.</p></div>
</div></div>
<div class="eli-subdivision" id="rct_12"><table><tbody><tr><td><p>(12)</p></td><td><p>Recital text.</p></td></tr></tbody></table></div>
"""

CONS_HTML = """
<div class="eli-subdivision" id="art_51">
<p class="title-article-norm">Article 51</p>
<div class="eli-title" id="art_51.tit_1"><p class="stitle-article-norm">Classification</p></div>
<div class="norm"><span class="no-parag">2.  </span><div class="norm inline-element">
Presumed when compute is greater than 10<span class="superscript">25</span>.</div></div>
<p class="modref"><a title="32026R1744: INSERTED">▼M1</a></p>
<div class="norm"><span class="no-parag">2a.  </span><div class="norm inline-element">New text.</div></div>
</div>
<div id="anx_III"><p class="title-annex-1">ANNEX III</p><p class="title-annex-2">High-risk AI systems</p>
<p class="norm">Areas:</p>
<div class="grid-container grid-list"><div class="list grid-list-column-1"><span>1. </span></div>
<div class="grid-list-column-2"><p class="norm">Biometrics</p></div></div>
<div class="grid-container grid-list"><div class="list grid-list-column-1"><span>2. </span></div>
<div class="grid-list-column-2"><p class="norm">Critical infrastructure</p></div></div>
</div>
"""


def soup(html):
    return BeautifulSoup(html, "lxml")


def test_oj_article_paragraphs_points_and_hierarchy():
    (art,) = parse_articles(soup(OJ_HTML), "AIA", "en")
    assert art.pid == "AIA:art_6"
    assert art.title == "Classification rules"
    assert art.chapter == "CHAPTER III — HIGH-RISK AI SYSTEMS"
    assert [u.num for u in art.units] == ["1", "2"]
    assert art.units[0].points == [("(a)", "safety component;"), ("(b)", "third-party assessment.")]
    assert "(a) safety component;" in art.units[0].text
    assert "(1)" not in art.units[1].text  # footnote marker removed


def test_consolidated_superscript_and_amendment_marker():
    (art,) = parse_articles(soup(CONS_HTML), "AIA", "en")
    assert "10^25" in art.units[0].text
    assert art.units[0].amended_by == []
    assert art.units[1].num == "2a" and art.units[1].amended_by == ["32026R1744"]
    assert art.amended_by == ["32026R1744"]
    assert "▼" not in art.text


def test_annex_units_and_title():
    (anx,) = parse_annexes(soup(CONS_HTML), "AIA", "en")
    assert anx.pid == "AIA:anx_III" and anx.title == "High-risk AI systems"
    assert [u.num for u in anx.units] == [None, "1", "2"]


def test_recitals():
    (r,) = parse_recitals(soup(OJ_HTML), "AIA", "en")
    assert r.pid == "AIA:rct_12" and r.text == "Recital text."
