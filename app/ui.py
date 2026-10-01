"""Minimal Streamlit front end for the API.   streamlit run app/ui.py   (needs `pip install -e ".[ui]"`)"""
import os

import httpx
import streamlit as st

API_URL = os.environ.get("MLRAG_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="mlrag", layout="centered")
st.title("Multilingual RAG")

with st.sidebar:
    mode = st.selectbox("Retrieval mode", ["hybrid", "dense", "sparse"])
    lang = st.selectbox("Restrict corpus language", ["any", "ne", "en"])
    top_k = st.slider("Passages", 1, 10, 5)

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

if question := st.chat_input("Ask in Nepali, Romanized Nepali or English"):
    with st.chat_message("user"):
        st.markdown(question)
    payload = {"question": question, "history": st.session_state.history, "mode": mode,
               "top_k": top_k, "lang": None if lang == "any" else lang}
    try:
        r = httpx.post(f"{API_URL}/ask", json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        st.error(f"API error: {e}")
    else:
        with st.chat_message("assistant"):
            st.markdown(data["answer"])
            if data["abstained"]:
                st.caption("The model abstained: the retrieved passages did not contain the answer.")
            st.caption(f"{data['query_tokens']} query tokens · {data['timings'].get('total_s', 0):.2f}s total")
            for i, h in enumerate(data["retrieved"], 1):
                cited = "✓ cited" if any(c["id"] == h["id"] for c in data["citations"]) else ""
                with st.expander(f"[{i}] {h['context']} ({h['lang']}) {cited}"):
                    st.write(h["text"])
                    st.markdown(f"[source]({h['url']})  ·  score {h['score']:.3f}")
        st.session_state.history += [{"role": "user", "content": question},
                                     {"role": "assistant", "content": data["answer"]}]
