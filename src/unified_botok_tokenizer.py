#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 13:47:23 2025

@author: dorje
"""

# unified_botok_tokenizer.py
# -*- coding: utf-8 -*-

"""
Unified Botok→spaCy tokenizer adapter with optional tsheg-splitting.

Usage from other scripts:
    from unified_botok_tokenizer import create_spacy_tokenizer_factory
    nlp = spacy.blank("xx")
    nlp.tokenizer = create_spacy_tokenizer_factory(
        config_path=None,                # or a path; default is hard-wired below
        force_split_tsheg=False          # set True to split 0F0B/0F0C as separate tokens
    )(nlp)

Notes
- If `botok` is available, we use it; otherwise, we fall back to a
  simple offset-preserving tokenizer (whitespace + keep tsheg/shad as single-char tokens).
- When `force_split_tsheg=True`, we postprocess tokens to ensure every
  U+0F0B / U+0F0C becomes its own token while preserving offsets.
"""

from typing import List, Tuple, Optional, Callable
import os as _os

# Default Botok config path: resolve to <repo_root>/pybo/dialect_packs
# Override by passing config_path to create_spacy_tokenizer_factory()
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DEFAULT_BOTOK_CONFIG_PATH = _os.path.join(_REPO_ROOT, "pybo", "dialect_packs")

# Tibetan punctuation we care about
TSEK = "\u0F0B"
NB_TSEK = "\u0F0C"
SHADS = {"\u0F0D", "\u0F0E"}  # shad, nyis-shad


# ---------- Botok-backed segmenter (if available) ----------
def _make_botok_segmenter(config_path: Optional[str]) -> Optional[Callable[[str], List[Tuple[int, int, str]]]]:
    try:
        # Botok import
        from botok import WordTokenizer  # type: ignore
    except Exception:
        return None

    # Choose config path (explicit > default > None)
    cfg = config_path if config_path is not None else DEFAULT_BOTOK_CONFIG_PATH

    try:
        wt = WordTokenizer(config=cfg)
    except Exception:
        # If config fails, try without config
        try:
            wt = WordTokenizer()
        except Exception:
            return None

    def segment(text: str) -> List[Tuple[int, int, str]]:
        """
        Return list of (start, end, token_text) using Botok tokens.
        We try a few attribute names to extract character offsets robustly.
        """
        toks: List[Tuple[int, int, str]] = []
        # Most common: wt.tokenize(text) -> list of token objects
        # We'll be defensive about field names.
        try:
            botok_tokens = wt.tokenize(text)
        except Exception:
            # As an extreme fallback, return empty, so caller can fallback
            return toks

        for tok in botok_tokens:
            # Candidate attribute names we’ve seen in various botok versions
            # We check in order and use the first that exists.
            start = None
            end = None
            token_text = None

            # 1) dict-like
            if hasattr(tok, "get"):
                try:
                    token_text = tok.get("text", None)
                    start = tok.get("start", tok.get("char_start", None))
                    end = tok.get("end", tok.get("char_end", None))
                except Exception:
                    pass

            # 2) attribute-like
            if token_text is None and hasattr(tok, "text"):
                try:
                    token_text = tok.text
                except Exception:
                    token_text = None
            if start is None and hasattr(tok, "start"):
                try:
                    start = getattr(tok, "start")
                except Exception:
                    start = None
            if end is None and hasattr(tok, "end"):
                try:
                    end = getattr(tok, "end")
                except Exception:
                    end = None

            # 3) char_span or similar tuple
            if (start is None or end is None) and hasattr(tok, "char_span"):
                try:
                    cs = getattr(tok, "char_span")
                    if isinstance(cs, (list, tuple)) and len(cs) == 2:
                        start, end = int(cs[0]), int(cs[1])
                except Exception:
                    pass

            # 4) last-resort: find substring from previous end (can be wrong if repeats);
            # we prefer not to do this, so only use if everything else failed.
            if (start is None or end is None) and token_text is not None:
                # naive scan; try to locate the next occurrence
                # this is a fallback and may be off in pathological cases
                prev_end = toks[-1][1] if toks else 0
                idx = text.find(token_text, prev_end)
                if idx >= 0:
                    start = idx
                    end = idx + len(token_text)

            if (
                token_text is not None
                and isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= len(text)
            ):
                toks.append((start, end, token_text))

        return toks

    return segment


# ---------- Simple fallback segmenter ----------
def _fallback_segmenter(text: str) -> List[Tuple[int, int, str]]:
    """
    Offset-preserving fallback:
    - Split on whitespace
    - Emit tsheg/shad as separate single-char tokens
    """
    out: List[Tuple[int, int, str]] = []
    i = 0
    L = len(text)
    start = None
    while i < L:
        ch = text[i]
        if ch.isspace():
            if start is not None:
                out.append((start, i, text[start:i]))
                start = None
            i += 1
            continue
        if ch == TSEK or ch == NB_TSEK or ch in SHADS:
            if start is not None:
                out.append((start, i, text[start:i]))
                start = None
            out.append((i, i + 1, ch))
            i += 1
            continue
        if start is None:
            start = i
        i += 1
    if start is not None:
        out.append((start, L, text[start:L]))
    return out


# ---------- Post-processor: force-split tsheg ----------
def _force_split_tsheg_tokens(tokens: List[Tuple[int, int, str]], text: str) -> List[Tuple[int, int, str]]:
    """
    Given tokens as (start, end, token_text), split any token containing
    U+0F0B or U+0F0C so that each tsheg becomes its own one-char token.
    Offsets are preserved exactly.
    """
    result: List[Tuple[int, int, str]] = []
    for (s, e, tok) in tokens:
        # Fast path: no tsheg inside
        if (TSEK not in tok) and (NB_TSEK not in tok):
            result.append((s, e, tok))
            continue

        # Walk the characters within this token and split at tshegs
        j = s
        while j < e:
            ch = text[j]
            if ch == TSEK or ch == NB_TSEK:
                # tsheg itself: single-char token
                result.append((j, j + 1, ch))
                j += 1
            else:
                # accumulate until next tsheg or end
                k = j + 1
                while k < e and text[k] not in (TSEK, NB_TSEK):
                    k += 1
                result.append((j, k, text[j:k]))
                j = k
    return result


def _fill_nonspace_gaps(tokens: List[Tuple[int, int, str]], text: str) -> List[Tuple[int, int, str]]:
    """
    Insert tokens for any skipped non-whitespace characters between Botok tokens.
    Keeps offsets exact; does NOT add pure whitespace (spaces/newlines),
    because spaCy's Doc(words, spaces) will handle whitespace via `spaces`.
    """
    if not tokens:
        return tokens
    tokens = sorted(tokens, key=lambda x: x[0])
    out: List[Tuple[int, int, str]] = []
    pos = 0
    for s, e, tok in tokens:
        if s > pos:
            gap = text[pos:s]
            if gap and not gap.isspace():
                out.append((pos, s, gap))
        out.append((s, e, tok if tok is not None else text[s:e]))
        pos = e
    if pos < len(text):
        tail = text[pos:]
        if tail and not tail.isspace():
            out.append((pos, len(text), tail))
    return out


# ---------- Public factory for spaCy ----------
def create_spacy_tokenizer_factory(
    config_path: Optional[str] = None,
    force_split_tsheg: bool = True,
):
    """
    Returns a factory that, given an `nlp`, produces a spaCy-compatible tokenizer.

    Parameters
    ----------
    config_path : Optional[str]
        Path to Botok config (if None, uses DEFAULT_BOTOK_CONFIG_PATH).
    force_split_tsheg : bool
        If True, split U+0F0B/U+0F0C into standalone tokens (post-process).
    """
    # Build the best segmenter we can (Botok if possible, else fallback)
    botok_segment = _make_botok_segmenter(config_path)
    segment_fn = botok_segment if botok_segment is not None else _fallback_segmenter

    def factory(nlp):
        from spacy.tokens import Doc

        class _Tokenizer:
            def __init__(self, vocab):
                self.vocab = vocab

            def __call__(self, text: str) -> "Doc":
                tokens = segment_fn(text)
                tokens = _fill_nonspace_gaps(tokens, text)  # ← new step to restore skipped chars

                if force_split_tsheg:
                    tokens = _force_split_tsheg_tokens(tokens, text)

                # Convert to spaCy Doc (words + spaces)
                words: List[str] = []
                spaces: List[bool] = []

                for i, (s, e, _) in enumerate(tokens):
                    words.append(text[s:e])
                    if i < len(tokens) - 1:
                        next_start = tokens[i + 1][0]
                        gap = text[e:next_start]
                        spaces.append(len(gap) > 0 and gap.isspace())
                    else:
                        spaces.append(False)

                return Doc(self.vocab, words=words, spaces=spaces)

        return _Tokenizer(nlp.vocab)

    return factory