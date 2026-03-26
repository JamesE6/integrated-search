"""
ner_processor.py – wraps SpacyNLPProcessor for Streamlit.

Handles:
    - Lazy model loading via st.cache_resource
    - Graceful degradation when model / botok unavailable
    - Single-text and CSV NER via the existing executor code
"""

import json
import csv
import io
import sys
import os
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Ensure the src/ directory is importable
_src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


def _model_available() -> bool:
    """Check whether the spaCy model and botok deps can be loaded."""
    try:
        import spacy  # noqa: F401
        from unified_botok_tokenizer import create_spacy_tokenizer_factory  # noqa: F401
        return True
    except Exception:
        return False


@st.cache_resource(show_spinner="Loading Tibetan NER model…")
def load_processor(model_name: str = "bo_core_news_lg"):
    """
    Load the SpacyNLPProcessor once and cache it.
    Returns (processor, None) on success or (None, error_message) on failure.
    """
    try:
        import subprocess
        import spacy
        from botok_loader import BoTokTokenizer  # noqa: F401
        from unified_botok_tokenizer import create_spacy_tokenizer_factory  # noqa: F401

        try:
            nlp = spacy.load(model_name)
        except OSError:
            # Model not pip-installed — try installing from local tarball
            tarball = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "packages",
                "bo_core_news_lg-0.0.7.tar.gz",
            )
            if os.path.exists(tarball):
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", tarball]
                )
                nlp = spacy.load(model_name)
            else:
                raise OSError(
                    f"Model '{model_name}' not installed and tarball not found at {tarball}"
                )

        class _Processor:
            """Lightweight wrapper matching the SpacyNLPProcessor interface."""

            def __init__(self, nlp_instance):
                self.nlp = nlp_instance

            def _configure_ner_tokenizer(self):
                self.nlp.tokenizer = create_spacy_tokenizer_factory(
                    force_split_tsheg=True
                )(self.nlp)

            def perform_ner(self, text: str) -> Dict[str, Any]:
                self._configure_ner_tokenizer()
                with self.nlp.select_pipes(enable=["ner"]):
                    doc = self.nlp(text)
                entities = [
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                    for ent in doc.ents
                ]
                return {
                    "input_text": text,
                    "entities": entities,
                    "entity_count": len(entities),
                }

            def perform_ner_on_csv(
                self,
                csv_text: str,
                content_col: str = "normalised_paragraph",
                id_col: str = "paragraph_idx",
                filename_col: str = "filename",
                year_col: str = "year",
                month_col: str = "month",
                day_col: str = "date",
                batch_size: int = 64,
            ) -> List[Dict[str, Any]]:
                """Run NER on CSV content (passed as a string, not a path)."""
                reader = csv.DictReader(io.StringIO(csv_text))
                if reader.fieldnames is None:
                    raise ValueError("CSV has no header row.")

                fn_map = {name.lower(): name for name in reader.fieldnames}

                def resolve(col):
                    key = col.lower()
                    if key not in fn_map:
                        raise ValueError(
                            f"Missing column '{col}'. Found: {reader.fieldnames}"
                        )
                    return fn_map[key]

                id_r = resolve(id_col)
                content_r = resolve(content_col)
                filename_r = resolve(filename_col)
                year_r = resolve(year_col)
                month_r = resolve(month_col)
                day_r = resolve(day_col)

                metas: List[Dict[str, str]] = []
                texts: List[str] = []
                for row in reader:
                    text = (row.get(content_r) or "").strip()
                    y = str(row.get(year_r, "")).strip() or "0000"
                    m = str(row.get(month_r, "")).strip() or "00"
                    d = str(row.get(day_r, "")).strip() or "00"
                    metas.append({
                        "Id": row.get(id_r),
                        "Filename": row.get(filename_r),
                        "Date": f"{y.zfill(4)}-{m.zfill(2)}-{d.zfill(2)}",
                    })
                    texts.append(text)

                self._configure_ner_tokenizer()
                results: List[Dict[str, Any]] = []
                with self.nlp.select_pipes(enable=["ner"]):
                    for meta, doc in zip(
                        metas, self.nlp.pipe(texts, batch_size=batch_size)
                    ):
                        ents = [
                            {
                                "text": ent.text,
                                "label": ent.label_,
                                "start": ent.start_char,
                                "end": ent.end_char,
                            }
                            for ent in doc.ents
                        ]
                        results.append({
                            "Id": meta["Id"],
                            "Filename": meta["Filename"],
                            "Date": meta["Date"],
                            "input_text": doc.text,
                            "entity_count": len(ents),
                            "entities": ents,
                        })
                return results

        return _Processor(nlp), None

    except Exception as exc:
        return None, str(exc)


def is_model_available() -> bool:
    return _model_available()
