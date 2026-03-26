"""
ner_display.py – rendering utilities for NER results in Streamlit.

Provides:
    - Inline entity highlighting (HTML spans over Tibetan text)
    - Label colour mapping
    - Summary statistics
    - Filtering helpers
"""

import html
from typing import Any, Dict, List, Optional

import pandas as pd


# ── Label colours ──────────────────────────────────────────────────────────
LABEL_COLOURS: Dict[str, str] = {
    "PER":      "#4a90d9",   # blue
    "ORG":      "#50b86c",   # green
    "LOC":      "#e07b4a",   # orange
    "TIME":     "#9b6fc3",   # purple
    "SOC":      "#3db8b8",   # teal
    "POSITION": "#c9a834",   # gold
    "TITLE":    "#d65fa0",   # pink
    "SLOGAN":   "#a07850",   # brown
    "EVENT":    "#e05555",   # coral
}

DEFAULT_COLOUR = "#888888"


def label_colour(label: str) -> str:
    return LABEL_COLOURS.get(label.upper(), DEFAULT_COLOUR)


def label_badge_html(label: str) -> str:
    """Small coloured badge for a label."""
    c = label_colour(label)
    return (
        f'<span style="background:{c};color:#fff;padding:1px 6px;'
        f'border-radius:3px;font-size:0.75em;vertical-align:middle;'
        f'margin-left:2px;">{html.escape(label)}</span>'
    )


# ── Inline entity highlighting ────────────────────────────────────────────
def render_annotated_text(
    text: str,
    entities: List[Dict[str, Any]],
    selected_labels: Optional[set] = None,
    translations: Optional[Dict[str, str]] = None,
) -> str:
    """
    Return an HTML string with entity spans highlighted inline.

    Parameters
    ----------
    text : str
        The original Tibetan input text.
    entities : list of dict
        Each dict must have 'start', 'end', 'label' (and optionally 'text').
    selected_labels : set or None
        If provided, only highlight entities whose label is in this set.
    translations : dict or None
        If provided, maps Tibetan entity text → English translation.
        Entities with a translation get a tooltip on hover.
    """
    # Sort entities by start offset; drop overlaps (keep earlier/longer).
    ents = sorted(entities, key=lambda e: (e["start"], -e["end"]))
    filtered: List[Dict[str, Any]] = []
    last_end = -1
    for e in ents:
        if selected_labels and e["label"] not in selected_labels:
            continue
        if e["start"] >= last_end:
            filtered.append(e)
            last_end = e["end"]

    parts: list[str] = []
    cursor = 0
    for e in filtered:
        s, end, label = e["start"], e["end"], e["label"]
        if s > cursor:
            parts.append(html.escape(text[cursor:s]))
        c = label_colour(label)
        span_text = html.escape(text[s:end])
        raw_ent_text = text[s:end]

        # Build tooltip if translation available
        tooltip_attr = ""
        if translations and raw_ent_text in translations:
            tip = html.escape(f"{translations[raw_ent_text]} ({label})")
            tooltip_attr = f' title="{tip}"'

        parts.append(
            f'<span style="background:{c}22;border-bottom:2px solid {c};'
            f'padding:1px 2px;border-radius:2px;cursor:default;"'
            f'{tooltip_attr}>'
            f'{span_text}{label_badge_html(label)}</span>'
        )
        cursor = end
    if cursor < len(text):
        parts.append(html.escape(text[cursor:]))

    return (
        '<div style="font-size:1.15em;line-height:1.9;direction:ltr;'
        'font-family:\'Jomolhari\',\'Noto Sans Tibetan\',\'Microsoft Himalaya\',sans-serif;">'
        + "".join(parts)
        + "</div>"
    )


# ── Legend ─────────────────────────────────────────────────────────────────
def render_label_legend(labels: Optional[set] = None) -> str:
    """Return HTML for a horizontal legend of entity labels."""
    show = labels if labels else set(LABEL_COLOURS.keys())
    items = []
    for lab in sorted(show):
        c = label_colour(lab)
        items.append(
            f'<span style="display:inline-block;margin:0 8px 4px 0;">'
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{c};border-radius:2px;vertical-align:middle;'
            f'margin-right:4px;"></span>'
            f'<span style="font-size:0.85em;">{html.escape(lab)}</span></span>'
        )
    return '<div style="margin-bottom:8px;">' + "".join(items) + "</div>"


# ── Summary statistics ────────────────────────────────────────────────────
def entity_summary_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Given a list of NER result records (each with 'entities'),
    return a DataFrame summarising label counts.
    """
    counts: Dict[str, int] = {}
    for rec in records:
        for ent in rec.get("entities", []):
            lab = ent["label"]
            counts[lab] = counts.get(lab, 0) + 1
    rows = [{"Label": k, "Count": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Label", "Count"])


def all_labels_in_records(records: List[Dict[str, Any]]) -> set:
    """Collect every distinct entity label across records."""
    labels: set = set()
    for rec in records:
        for ent in rec.get("entities", []):
            labels.add(ent["label"])
    return labels


# ── Entity table ──────────────────────────────────────────────────────────
def entities_to_df(
    records: List[Dict[str, Any]],
    selected_labels: Optional[set] = None,
) -> pd.DataFrame:
    """Flatten all entities across records into a single DataFrame."""
    rows = []
    for rec in records:
        for ent in rec.get("entities", []):
            if selected_labels and ent["label"] not in selected_labels:
                continue
            rows.append({
                "Entity": ent["text"],
                "Label": ent["label"],
                "Paragraph ID": rec.get("Id", ""),
                "Date": rec.get("Date", ""),
                "Filename": rec.get("Filename", ""),
                "Start": ent["start"],
                "End": ent["end"],
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Entity", "Label", "Paragraph ID", "Date", "Filename", "Start", "End"]
    )


# ── Unique entities for translation ───────────────────────────────────────
def unique_entities(
    records: List[Dict[str, Any]],
    selected_labels: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    Return deduplicated entities as [{"text": ..., "label": ..., "count": ...}],
    sorted by count descending.
    """
    counts: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        for ent in rec.get("entities", []):
            if selected_labels and ent["label"] not in selected_labels:
                continue
            key = (ent["text"], ent["label"])
            if key not in counts:
                counts[key] = {"text": ent["text"], "label": ent["label"], "count": 0}
            counts[key]["count"] += 1
    return sorted(counts.values(), key=lambda x: -x["count"])
