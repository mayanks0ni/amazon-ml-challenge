"""Legal form extraction and canonicalization (Section 10.1)."""
import re
from typing import Optional, Set, Tuple

# Canonical legal form categories
LEGAL_FORM_PATTERNS = [
    # Private limited variants
    (r"\b(pvt\.?\s*ltd\.?|private\s+limited|p\.?\s*ltd\.?|praivet\s+limited|praivrr\s+limirrd|pra\.?\s*li\.?)\b", "PRIVATE_LIMITED"),
    # Public limited / limited variants
    (r"\b(ltd\.?|limited|public\s+limited|limirrd)\b", "LIMITED"),
    # Corporation
    (r"\b(corp\.?|corporation)\b", "CORPORATION"),
    # Incorporated
    (r"\b(inc\.?|incorporated)\b", "INCORPORATED"),
    # LLC
    (r"\b(l\.?l\.?c\.?|llc|limited\s+liability\s+company)\b", "LLC"),
    # LLP
    (r"\b(l\.?l\.?p\.?|llp|limited\s+liability\s+partnership|elelpi)\b", "LLP"),
    # Partnership / Proprietorship
    (r"\b(proprietorship|prop\.?)\b", "PROPRIETORSHIP"),
    (r"\b(partnership|partn\.?)\b", "PARTNERSHIP"),
    # European / French forms
    (r"\b(s\.?a\.?r\.?l\.?|sarl)\b", "SARL"),
    (r"\b(s\.?a\.?s\.?|sas)\b", "SAS"),
    (r"\b(s\.?a\.?|sa)\b", "SA"),
    (r"\b(gmbh)\b", "GMBH"),
    # Company / Enterprise / Holdings
    (r"\b(co\.?|company)\b", "COMPANY"),
    (r"\b(enterprises?|ents?\.?)\b", "ENTERPRISE"),
    (r"\b(holdings?)\b", "HOLDINGS"),
]


def extract_legal_form(name_norm: str) -> Tuple[str, str]:
    """Extracts canonical legal form from normalized name and returns (name_core, legal_form).

    The legal form is stripped from name_norm to form name_core.
    """
    if not name_norm:
        return "", ""

    found_form = ""
    core = name_norm.lower()

    for pattern, canonical in LEGAL_FORM_PATTERNS:
        match = re.search(pattern, core, re.IGNORECASE)
        if match:
            found_form = canonical
            # Remove the legal form phrase from core
            core = re.sub(pattern, " ", core, flags=re.IGNORECASE)
            break

    # Clean up whitespace and dangling punctuation
    core = re.sub(r"\s+", " ", core).strip(" ,.-/")
    return core, found_form


def check_legal_form_compatibility(form_a: str, form_b: str) -> Tuple[int, int]:
    """Returns (legal_form_eq, legal_form_conflict).

    - legal_form_eq: 1 if both non-empty and equal, 0 otherwise.
    - legal_form_conflict: 1 if both non-empty and different, 0 otherwise.
    """
    if not form_a or not form_b:
        return 0, 0
    if form_a == form_b:
        return 1, 0
    return 0, 1
