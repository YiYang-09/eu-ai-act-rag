"""Streamlit UI for the EU AI Act & GDPR assistant.

Two modes:
  Live    – calls the FastAPI service (API_URL, default http://localhost:8000).
  Replay  – browses the 94 benchmark questions with the answers qwen2.5:7b gave on
            graph+rerank contexts, scored against the answer keys. Needs nothing running.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
REPLAY = Path(__file__).with_name("replay.json")
ROLES = {"public": "Public (AI Act + GDPR)", "dpo": "Data protection officer (GDPR only)",
         "ai-governance": "AI governance (AI Act only)"}

st.set_page_config(page_title="EU AI Act & GDPR assistant", page_icon="⚖️", layout="wide")


@st.cache_data
def load_replay() -> list[dict]:
    return json.loads(REPLAY.read_text(encoding="utf-8")) if REPLAY.exists() else []


def api_ok() -> dict | None:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).json()
    except requests.RequestException:
        return None


def show_answer(answer: str, insufficient: bool, citations: list[dict]) -> None:
    if insufficient:
        st.warning(answer or "The sources do not answer this question.")
    else:
        st.markdown(answer)
    cited = [c for c in citations if c["cited"]]
    if cited:
        st.caption("Cited: " + " · ".join(f"[{c['n']}] [{c['label']}]({c['url']})" for c in cited))
    st.subheader("Sources")
    for c in citations:
        mark = "✅ cited" if c["cited"] else "retrieved"
        via = f" · {c['via']}" if c.get("via") and c["via"] != "retrieval" else ""
        with st.expander(f"[{c['n']}] {c['label']} ({mark}{via})"):
            st.caption(c["header"])
            st.write(c["text"])
            st.markdown(f"[Open on EUR-Lex]({c['url']})")


st.title("⚖️ EU AI Act & GDPR compliance assistant")
st.caption("Answers only from the law text (AI Act as amended by Regulation (EU) 2026/1744, GDPR), "
           "with numbered citations, and says so when the texts do not answer.")

health = api_ok()
mode = st.sidebar.radio("Mode", ["Live", "Replay"], index=0 if health else 1,
                        help="Replay shows recorded benchmark answers and needs no GPU.")
if health:
    st.sidebar.success(f"API up · LLM {health['llm']} ({'reachable' if health['llm_reachable'] else 'unreachable'})")
else:
    st.sidebar.info(f"API not reachable at {API_URL}; replay mode only.")

if mode == "Live":
    role = st.sidebar.selectbox("Role (document access)", list(ROLES), format_func=ROLES.get)
    lang = st.sidebar.selectbox("Language", ["auto", "en", "sv"])
    q = st.text_input("Question", placeholder="When do the obligations for high-risk AI systems in Annex III apply?")
    if st.button("Ask", type="primary", disabled=not (health and q)):
        with st.spinner("Retrieving and answering…"):
            r = requests.post(f"{API_URL}/ask", json={"question": q, "role": role,
                                                     "lang": None if lang == "auto" else lang}, timeout=300)
        if r.status_code != 200:
            st.error(r.text)
        else:
            a = r.json()
            show_answer(a["answer"], a["insufficient"], a["citations"])
            st.caption(f"model {a['model']} · timings {a['timings_ms']} ms · tokens {a['tokens']}")
else:
    rows = load_replay()
    if not rows:
        st.error("app/replay.json missing; run scripts/build_replay.py")
        st.stop()
    cats = sorted({r["category"] for r in rows})
    pick_cat = st.sidebar.multiselect("Category", cats)
    pick_lang = st.sidebar.multiselect("Language", ["en", "sv"])
    only_wrong = st.sidebar.checkbox("Only failures")
    view = [r for r in rows if (not pick_cat or r["category"] in pick_cat) and (not pick_lang or r["lang"] in pick_lang)
            and (not only_wrong or r["correct"] is False or (not r["answerable"] and not r["refused"]))]
    st.sidebar.caption(f"{len(view)} of {len(rows)} questions")
    if not view:
        st.stop()
    r = st.selectbox("Benchmark question", view, format_func=lambda r: f"{r['id']} · {r['question']}")
    st.markdown(f"**{r['question']}**")
    if r["answerable"]:
        verdict = "✅ correct" if r["correct"] else "❌ wrong"
    else:
        verdict = "✅ refused (unanswerable)" if r["refused"] else "❌ answered an unanswerable question"
    st.caption(f"{r['category']} · {r['lang']} · {r['model']} · {r['latency_s']} s · {verdict}")
    show_answer(r["answer"], r["insufficient"], r["citations"])
    with st.expander("Reference answer"):
        st.write(r["reference"] or "(unanswerable from the corpus)")
