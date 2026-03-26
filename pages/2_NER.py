"""
2_NER.py – Named Entity Recognition: run live or browse pre-computed results.
"""

import json
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd

from utils.ner_display import (
    render_annotated_text,
    render_label_legend,
    entity_summary_df,
    all_labels_in_records,
    entities_to_df,
    unique_entities,
    LABEL_COLOURS,
)
from utils.translator import translate_bo_en, is_translation_available

st.set_page_config(page_title="Tibetan NER", page_icon="🏷️", layout="wide")

st.title("🏷️ Named Entity Recognition")
st.caption("Run the Tibetan NER model or browse pre-computed results")


# ── Check model availability & preload ────────────────────────────────────
@st.cache_data(show_spinner=False)
def check_model():
    from utils.ner_processor import is_model_available
    return is_model_available()


model_ok = check_model()

# If the model is available, preload it on first visit so the user sees progress
if model_ok:
    from utils.ner_processor import load_processor

    @st.cache_resource(show_spinner=False)
    def _preload():
        return load_processor()

    if "_model_loaded" not in st.session_state:
        with st.status("Loading Tibetan NER model…", expanded=True) as status:
            st.write("Initializing spaCy pipeline…")
            st.write("Loading bo_core_news_lg model weights…")
            st.write("Configuring botok tokenizer…")
            proc, err = _preload()
            if err:
                status.update(label="Model failed to load", state="error", expanded=True)
                st.error(f"Error: {err}")
                model_ok = False
            else:
                status.update(label="Model loaded ✓", state="complete", expanded=False)
        st.session_state["_model_loaded"] = True


# ── Sidebar mode selection ────────────────────────────────────────────────
st.sidebar.header("Mode")

modes = ["Browse Results (JSON)", "Run NER (live)"]
if not model_ok:
    modes = ["Browse Results (JSON)", "Run NER (live) ⚠️ model unavailable"]

mode = st.sidebar.radio("Select mode:", modes, index=0)
is_live_mode = "Run NER" in mode and "unavailable" not in mode


# ═══════════════════════════════════════════════════════════════════════════
# BROWSE MODE
# ═══════════════════════════════════════════════════════════════════════════
def browse_mode():
    st.subheader("Browse pre-computed NER results")

    uploaded = st.file_uploader(
        "Upload a `_ner.json` file (output of the NER pipeline):",
        type=["json"],
        key="browse_upload",
    )

    # Also allow pasting JSON
    with st.expander("Or paste JSON directly"):
        pasted = st.text_area(
            "Paste NER JSON here:",
            height=150,
            key="browse_paste",
        )

    records: Optional[List[Dict[str, Any]]] = None

    if uploaded is not None:
        try:
            records = json.load(uploaded)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
    elif pasted and pasted.strip():
        try:
            records = json.loads(pasted)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")

    if records is None:
        st.info("Upload or paste a NER JSON file to get started.")
        return

    # Normalise: accept a single dict as a one-element list
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or not records:
        st.error("Expected a JSON array of NER result objects.")
        return

    _render_results(records)


# ═══════════════════════════════════════════════════════════════════════════
# LIVE NER MODE
# ═══════════════════════════════════════════════════════════════════════════
def live_ner_mode():
    if not model_ok:
        st.warning(
            "The NER model is not available in this environment. "
            "This usually means `bo_core_news_lg` or `botok` could not be loaded. "
            "Please use **Browse Results** mode instead, or deploy to a server "
            "with the model installed."
        )
        return

    st.subheader("Run NER on Tibetan text")

    input_type = st.radio(
        "Input type:",
        ["Paste text", "Upload text file", "Upload CSV"],
        horizontal=True,
    )

    records: Optional[List[Dict[str, Any]]] = None

    if input_type == "Paste text":
        text = st.text_area(
            "Enter Tibetan text:",
            height=150,
            placeholder="བཀྲ་ཤིས་ལྷུན་པོ་དགོན་པ་ཤི་ག་རྩེ་རྫོང་ཁུལ་དུ་ཡོད།",
        )
        if st.button("Run NER", type="primary") and text.strip():
            from utils.ner_processor import load_processor
            processor, err = load_processor()  # cached — instant after preload
            if err:
                st.error(f"Model error: {err}")
                return
            with st.spinner("Running NER…"):
                result = processor.perform_ner(text.strip())
            records = [result]

    elif input_type == "Upload text file":
        uploaded = st.file_uploader("Upload a .txt file:", type=["txt"], key="live_txt")
        if uploaded is not None:
            text = uploaded.read().decode("utf-8").strip()
            if st.button("Run NER", type="primary") and text:
                from utils.ner_processor import load_processor
                processor, err = load_processor()
                if err:
                    st.error(f"Model error: {err}")
                    return
                with st.spinner("Running NER…"):
                    result = processor.perform_ner(text)
                records = [result]

    elif input_type == "Upload CSV":
        st.markdown(
            "Expected columns: `normalised_paragraph`, `paragraph_idx`, "
            "`filename`, `year`, `month`, `date`"
        )
        uploaded = st.file_uploader("Upload a CSV file:", type=["csv"], key="live_csv")
        if uploaded is not None:
            csv_text = uploaded.read().decode("utf-8")
            batch_size = st.number_input("Batch size", min_value=1, value=64)
            if st.button("Run NER on CSV", type="primary"):
                from utils.ner_processor import load_processor
                processor, err = load_processor()
                if err:
                    st.error(f"Model error: {err}")
                    return
                with st.spinner("Processing CSV — this may take a moment…"):
                    try:
                        records = processor.perform_ner_on_csv(
                            csv_text, batch_size=batch_size
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                        return

    if records is None:
        return

    # Offer JSON download before rendering
    json_str = json.dumps(records, ensure_ascii=False, indent=2)
    st.download_button(
        "⬇ Download NER results (JSON)",
        data=json_str,
        file_name="ner_results.json",
        mime="application/json",
    )

    st.markdown("---")
    _render_results(records)


# ═══════════════════════════════════════════════════════════════════════════
# SHARED RESULT RENDERER
# ═══════════════════════════════════════════════════════════════════════════
def _render_results(records: List[Dict[str, Any]]):
    """Render NER results with filtering, highlighting, and tables."""

    # ── Sidebar: label filter ─────────────────────────────────────────────
    all_labels = all_labels_in_records(records)
    if not all_labels:
        st.info("No entities found in these results.")
        return

    st.sidebar.header("Entity Filters")
    selected_labels = set(
        st.sidebar.multiselect(
            "Show labels:",
            sorted(all_labels),
            default=sorted(all_labels),
        )
    )

    # ── Summary tab vs. paragraph view ────────────────────────────────────
    tab_paras, tab_summary, tab_table, tab_translated = st.tabs(
        ["📄 Paragraph View", "📊 Summary", "📋 Entity Table", "🌐 Translated Entities"]
    )

    # ── Build translations (shared across tabs) ───────────────────────────
    # Cached per set of unique entity texts so we don't re-translate on rerun
    translations: Optional[dict] = None
    if is_translation_available():
        uniq = unique_entities(records, selected_labels)
        if uniq and st.sidebar.checkbox("Translate entities", value=False):
            @st.cache_data(show_spinner="Translating entities…")
            def _translate_entities(texts: tuple) -> dict:
                return {t: translate_bo_en(t) for t in texts}

            entity_texts = tuple(e["text"] for e in uniq)
            translations = _translate_entities(entity_texts)

    # ── Tab 1: Paragraph-by-paragraph view ────────────────────────────────
    with tab_paras:
        st.markdown(render_label_legend(selected_labels), unsafe_allow_html=True)

        # Pagination
        total = len(records)
        per_page = st.select_slider(
            "Paragraphs per page:",
            options=[5, 10, 25, 50, 100],
            value=10,
        )
        n_pages = max(1, -(-total // per_page))  # ceil division
        page = st.number_input(
            f"Page (1–{n_pages}):",
            min_value=1,
            max_value=n_pages,
            value=1,
        )
        start = (page - 1) * per_page
        end = min(start + per_page, total)

        for i, rec in enumerate(records[start:end], start=start + 1):
            text = rec.get("input_text", "")
            entities = rec.get("entities", [])
            meta_parts = []
            if rec.get("Id"):
                meta_parts.append(f"**ID:** {rec['Id']}")
            if rec.get("Date") and rec["Date"] != "0000-00-00":
                meta_parts.append(f"**Date:** {rec['Date']}")
            if rec.get("Filename"):
                meta_parts.append(f"**File:** {rec['Filename']}")
            meta_parts.append(f"**Entities:** {rec.get('entity_count', len(entities))}")

            with st.container():
                st.markdown(f"##### Paragraph {i}")
                if meta_parts:
                    st.markdown(" · ".join(meta_parts))

                anno_html = render_annotated_text(
                    text, entities, selected_labels, translations
                )
                st.markdown(anno_html, unsafe_allow_html=True)

                # Optional full-paragraph translation
                if is_translation_available():
                    with st.expander("Translate paragraph"):
                        st.write(translate_bo_en(text))

                st.markdown("---")

    # ── Tab 2: Summary statistics ─────────────────────────────────────────
    with tab_summary:
        col_chart, col_table = st.columns([2, 1])

        summary_df = entity_summary_df(records)
        if not summary_df.empty:
            with col_table:
                st.markdown("**Entity counts by label**")
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                st.metric("Total paragraphs", len(records))
                st.metric(
                    "Paragraphs with entities",
                    sum(1 for r in records if r.get("entity_count", 0) > 0),
                )
                st.metric("Total entities", summary_df["Count"].sum())

            with col_chart:
                # Colour-coded bar chart
                chart_df = summary_df.copy()
                chart_df["Colour"] = chart_df["Label"].map(
                    lambda l: LABEL_COLOURS.get(l, "#888")
                )
                st.bar_chart(
                    chart_df.set_index("Label")["Count"],
                    color="#4a90d9",
                )

        # Date distribution if dates exist
        dates = [
            r["Date"]
            for r in records
            if r.get("Date") and r["Date"] != "0000-00-00"
        ]
        if dates:
            st.markdown("**Entity distribution by date**")
            date_counts: dict = {}
            for r in records:
                d = r.get("Date", "")
                if d and d != "0000-00-00":
                    date_counts[d] = date_counts.get(d, 0) + r.get("entity_count", 0)
            if date_counts:
                ddf = pd.DataFrame(
                    [{"Date": k, "Entities": v} for k, v in sorted(date_counts.items())]
                )
                st.bar_chart(ddf.set_index("Date"))

    # ── Tab 3: Full entity table ──────────────────────────────────────────
    with tab_table:
        ent_df = entities_to_df(records, selected_labels)
        if ent_df.empty:
            st.info("No entities match the selected labels.")
        else:
            st.markdown(f"**{len(ent_df)} entities** matching selected labels")

            # Search within entities
            search = st.text_input("Filter entities by text:", key="ent_search")
            if search:
                mask = ent_df["Entity"].str.contains(search, case=False, na=False)
                ent_df = ent_df[mask]

            st.dataframe(ent_df, use_container_width=True, hide_index=True)

            # CSV download
            csv_data = ent_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download entity table (CSV)",
                data=csv_data,
                file_name="entities.csv",
                mime="text/csv",
            )

    # ── Tab 4: Translated entities ────────────────────────────────────────
    with tab_translated:
        if not is_translation_available():
            st.warning(
                "Translation unavailable — set `AZURE_API_KEY` in secrets to enable."
            )
        else:
            uniq = unique_entities(records, selected_labels)
            if not uniq:
                st.info("No entities match the selected labels.")
            else:
                st.markdown(
                    f"**{len(uniq)} unique entities** — translating via Azure Translator"
                )

                @st.cache_data(show_spinner="Translating entities…")
                def _translate_for_table(texts: tuple) -> list:
                    return [translate_bo_en(t) for t in texts]

                texts = tuple(e["text"] for e in uniq)
                translated = _translate_for_table(texts)

                rows = []
                for ent, eng in zip(uniq, translated):
                    c = LABEL_COLOURS.get(ent["label"], "#888")
                    rows.append({
                        "Tibetan": ent["text"],
                        "English": eng,
                        "Category": ent["label"],
                        "Count": ent["count"],
                    })

                trans_df = pd.DataFrame(rows)

                # Search / filter
                search = st.text_input(
                    "Filter by Tibetan or English text:",
                    key="trans_search",
                )
                if search:
                    mask = (
                        trans_df["Tibetan"].str.contains(search, case=False, na=False)
                        | trans_df["English"].str.contains(search, case=False, na=False)
                    )
                    trans_df = trans_df[mask]

                st.dataframe(trans_df, use_container_width=True, hide_index=True)

                # CSV download
                csv_data = trans_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Download translated entities (CSV)",
                    data=csv_data,
                    file_name="entities_translated.csv",
                    mime="text/csv",
                )


# ═══════════════════════════════════════════════════════════════════════════
# DISPATCH
# ═══════════════════════════════════════════════════════════════════════════
if is_live_mode:
    live_ner_mode()
else:
    browse_mode()
