"""
translator.py – Azure Cognitive Services bo→en translation.

Reads credentials from st.secrets or environment variables.
"""

import uuid
from typing import Optional

import requests
import streamlit as st

ENDPOINT = "https://api.cognitive.microsofttranslator.com"
PATH = "/translate"
API_VERSION = "3.0"


def _get_key() -> Optional[str]:
    try:
        return st.secrets["AZURE_API_KEY"]
    except Exception:
        import os
        return os.environ.get("AZURE_API_KEY")


def _get_region() -> str:
    try:
        return st.secrets.get("AZURE_REGION", "uksouth")
    except Exception:
        import os
        return os.environ.get("AZURE_REGION", "uksouth")


def translate_bo_en(text: str) -> str:
    """Translate Tibetan text to English via Azure. Returns original on failure."""
    key = _get_key()
    if not key:
        return "[Translation unavailable — no API key]"

    try:
        resp = requests.post(
            f"{ENDPOINT}{PATH}",
            params={"api-version": API_VERSION, "from": "bo", "to": "en"},
            headers={
                "Ocp-Apim-Subscription-Key": key,
                "Ocp-Apim-Subscription-Region": _get_region(),
                "Content-type": "application/json",
                "X-ClientTraceId": str(uuid.uuid4()),
            },
            json=[{"text": text}],
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()[0]["translations"][0]["text"]
    except Exception as exc:
        return f"[Translation error: {exc}]"


def is_translation_available() -> bool:
    return _get_key() is not None
