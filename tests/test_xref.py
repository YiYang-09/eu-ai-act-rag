from lexrag.corpus.xref import extract_xrefs


def test_english_ranges_and_external_acts():
    text = ("Systems in Annex III shall comply with Articles 9 to 15 and Article 22(1) of Regulation (EU) "
            "2016/679 and Article 10 of Directive (EU) 2016/680.")
    refs = extract_xrefs(text, "en", "AIA")
    assert {"AIA:anx_III", "AIA:art_9", "AIA:art_15", "GDPR:art_22", "GDPR"} <= set(refs)
    assert "AIA:art_10" in refs  # from the 9..15 range, not from the Directive reference
    assert not any(r.endswith("art_22") and r.startswith("AIA") for r in refs)


def test_swedish_matches_english():
    sv = ("AI-system som avses i bilaga III ska uppfylla artiklarna 9–15 och artikel 22.1 i förordning (EU) "
          "2016/679 och artikel 10 i direktiv (EU) 2016/680.")
    en = ("Systems in Annex III shall comply with Articles 9 to 15 and Article 22(1) of Regulation (EU) "
          "2016/679 and Article 10 of Directive (EU) 2016/680.")
    assert extract_xrefs(sv, "sv", "AIA") == extract_xrefs(en, "en", "AIA")


def test_self_reference_is_dropped():
    assert extract_xrefs("as set out in Article 6(3)", "en", "AIA", self_id="AIA:art_6") == []
