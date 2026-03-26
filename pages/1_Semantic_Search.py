"""
1_Semantic_Search.py – Pinecone + Cohere semantic search over Tibetan texts.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Semantic Search", page_icon="🔍", layout="wide")

st.title("🔍 Semantic Search")
st.caption("Search Tibetan newspaper texts using multilingual embeddings")

# ── Credentials ────────────────────────────────────────────────────────────
def _secret(key: str, fallback: str = ""):
    try:
        return st.secrets[key]
    except Exception:
        import os
        return os.environ.get(key, fallback)


cohere_key = _secret("COHERE_API_KEY")
pinecone_key = _secret("PINECONE_API_KEY")

if not cohere_key or not pinecone_key:
    st.error(
        "Missing API keys. Set `COHERE_API_KEY` and `PINECONE_API_KEY` in "
        "`.streamlit/secrets.toml` or as environment variables."
    )
    st.stop()


# ── Clients (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def get_cohere():
    import cohere
    return cohere.Client(cohere_key)


@st.cache_resource
def get_pinecone_index():
    from pinecone import Pinecone
    pc = Pinecone(api_key=pinecone_key)
    return pc.Index("diverge-test")


co = get_cohere()
index = get_pinecone_index()


# ── Translation ───────────────────────────────────────────────────────────
from utils.translator import translate_bo_en, is_translation_available  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────
def normalize_matches(matches):
    rows = []
    for m in matches:
        if not isinstance(m, dict):
            md = getattr(m, "metadata", {}) or {}
            row = {
                **md,
                "ID": getattr(m, "id", None),
                "Score": getattr(m, "score", None),
            }
        else:
            md = m.get("metadata", {}) or {}
            row = {**md, "ID": m.get("id"), "Score": m.get("score")}
        rows.append(row)
    return rows


def render_results_table(matches, translate=False):
    data = normalize_matches(matches)
    df = pd.DataFrame(data)
    if translate:
        if "text" in df.columns:
            df["Translated Text"] = df["text"].apply(translate_bo_en)
        if "title" in df.columns:
            df["Translated Title"] = df["title"].apply(translate_bo_en)
    return df


def index_query(input_string, top_k=25, filters=None):
    xq = co.embed(
        texts=[input_string],
        model="embed-multilingual-v2.0",
        input_type="search_query",
        truncate="END",
    ).embeddings[0]

    query_params = {
        "vector": xq,
        "top_k": top_k,
        "include_metadata": True,
    }
    if filters:
        query_params["filter"] = filters
    return index.query(**query_params)


# ── Sidebar filters ───────────────────────────────────────────────────────
st.sidebar.header("Filters")

publications = ["All", "Tibet Daily"]
selected_publication = st.sidebar.selectbox("Publication:", publications)

selected_years = st.sidebar.slider("Year range:", 2020, 2023, (2021, 2022))
selected_months = st.sidebar.slider("Month range:", 1, 12, (1, 12))

filters = {}
if selected_publication != "All":
    filters["publication"] = {"$eq": selected_publication}
filters["year"] = {"$in": [str(i) for i in range(selected_years[0], selected_years[1] + 1)]}
filters["month"] = {"$in": [str(i) for i in range(selected_months[0], selected_months[1] + 1)]}


# ── Main search interface ─────────────────────────────────────────────────
input_text = st.text_input("Enter your query (Tibetan or English):")

if input_text:
    translate_query = st.checkbox("Show English translation of query", value=False)
    if translate_query and is_translation_available():
        st.info(f"Translation: {translate_bo_en(input_text)}")

    num_results = st.slider("Number of results", min_value=1, max_value=100, value=10)
    pretty_print = st.checkbox("Formatted table view", value=True)

    with st.spinner("Searching…"):
        results = index_query(input_text, top_k=num_results, filters=filters)
    matches = results.matches

    if not matches:
        st.warning("No results found for this query and filter combination.")
    elif pretty_print:
        translator = st.checkbox("Translate results to English", value=False)
        df = render_results_table(matches, translate=translator)
        st.dataframe(df, use_container_width=True)
    else:
        st.json(
            results.to_dict()
            if hasattr(results, "to_dict")
            else {"matches": normalize_matches(matches)}
        )
