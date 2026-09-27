"""Landmark phrase extraction (Section 10.2)."""
import re
from typing import Tuple

LANDMARK_PREFIXES = [
    r"\b(?:near|opp\.?|opposite|behind|beside|next\s+to|adjacent\s+to)\b",
]

LANDMARK_RE = re.compile(
    r"\b(?:near|opp\.?|opposite|behind|beside|next\s+to|adjacent\s+to)\s+([^,;]+)",
    re.IGNORECASE,
)


def extract_landmark_phrase(addr_norm: str) -> Tuple[str, str]:
    """Extracts landmark phrase from normalized address and returns (cleaned_addr, landmark).

    Landmark is excluded from core address similarity so landmark differences
    do not distort street matches.
    """
    if not addr_norm:
        return "", ""

    match = LANDMARK_RE.search(addr_norm)
    if not match:
        return addr_norm, ""

    landmark = match.group(1).strip()
    # Remove landmark segment from the address text
    start, end = match.span()
    cleaned = addr_norm[:start] + " " + addr_norm[end:]
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-/")

    return cleaned, landmark
