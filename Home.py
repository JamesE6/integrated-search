"""
Home.py – Landing page for the Divergent Discourses Tibetan NLP toolkit.
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Tibetan NLP Toolkit",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Auto-initialize botok on first run ────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _ensure_botok():
    """Check for custom dialect pack; run setup_botok if missing."""
    repo_root = Path(__file__).resolve().parent
    custom_dir = repo_root / "pybo" / "dialect_packs" / "custom"
    dict_src = repo_root / "packages" / "modern-botok" / "dictionary" / "tsikchen.tsv"

    if custom_dir.exists():
        return True, None  # already set up

    if not dict_src.exists():
        return False, (
            "Cannot initialize botok: custom dictionary not found at "
            f"`{dict_src}`. Ensure `packages/modern-botok/` is present."
        )

    try:
        import shutil
        from botok.config import Config
        from botok import WordTokenizer

        dialect_packs = repo_root / "pybo" / "dialect_packs"
        dialect_packs.mkdir(parents=True, exist_ok=True)

        # Generate default 'general' dialect pack
        config = Config(base_path=str(dialect_packs))
        wt = WordTokenizer(config=config)

        # Copy general → custom
        general_dir = dialect_packs / "general"
        shutil.copytree(general_dir, custom_dir)

        # Replace dictionary
        dict_dest = custom_dir / "dictionary" / "words" / "tsikchen.tsv"
        dict_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dict_src, dict_dest)

        # Regenerate trie
        config = Config(dialect_name="custom", base_path=str(dialect_packs))
        wt = WordTokenizer(config=config)

        return True, None
    except Exception as exc:
        return False, str(exc)


if "_botok_checked" not in st.session_state:
    with st.status("Checking botok tokenizer…", expanded=True) as status:
        st.write("Looking for custom dialect pack…")
        ok, err = _ensure_botok()
        if ok and err is None:
            status.update(label="Botok ready ✓", state="complete", expanded=False)
        elif ok:
            status.update(label="Botok ready ✓", state="complete", expanded=False)
        else:
            st.write("Generating default dialect pack…")
            st.write("Installing custom dictionary…")
            st.write("Regenerating trie…")
            # _ensure_botok already ran and failed — show the error
            status.update(label="Botok setup failed", state="error", expanded=True)
            st.error(err)
    st.session_state["_botok_checked"] = True

st.title("📜 Tibetan NLP Toolkit")
st.caption("Divergent Discourses Project")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Semantic Search")
    st.markdown(
        "Search a Pinecone-indexed corpus of Tibetan newspaper texts using "
        "multilingual embeddings (Cohere). Filter by publication, date range, "
        "and optionally translate results to English."
    )
    st.page_link("pages/1_Semantic_Search.py", label="Go to Semantic Search →")

with col2:
    st.subheader("🏷️ Named Entity Recognition")
    st.markdown(
        "Run the Tibetan NER model on raw text or CSV files, or browse "
        "pre-computed NER results. View entities highlighted inline with "
        "the Tibetan source text, filter by label, and export results."
    )
    st.page_link("pages/2_NER.py", label="Go to NER →")

st.markdown("---")

with st.expander("About this project"):
    st.markdown(
        "This toolkit was developed as part of the "
        "[Divergent Discourses](https://github.com/Divergent-Discourses) project. "
        "The NER pipeline is built on spaCy with a custom `bo_core_news_lg` model "
        "and botok-based tokenization, trained on 1950s–1960s Tibetan newspaper data.\n\n"
        "**NER tag set:** PER · ORG · LOC · TIME · SOC · POSITION · TITLE · SLOGAN · EVENT\n\n"
        "The semantic search system uses Cohere multilingual embeddings indexed in Pinecone, "
        "with optional Tibetan→English translation via Azure Cognitive Services."
    )
