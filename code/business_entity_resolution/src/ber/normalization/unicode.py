"""Unicode normalization utilities using Python stdlib unicodedata."""
import unicodedata
import re


def normalize_unicode(text: str) -> str:
    """Applies NFKC normalization, lowercases, and strips surrounding whitespace."""
    if not text:
        return ""
    # NFKC normalizes compatibility characters (ligatures, fullwidth chars, etc.)
    text = unicodedata.normalize("NFKC", text)
    return text.lower().strip()


def strip_accents(text: str) -> str:
    """Strips diacritical marks/accents using NFKD decomposition.
    e.g., 'café' -> 'cafe', 'Société' -> 'societe'.
    Completely avoids GPL Unidecode library.
    """
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
