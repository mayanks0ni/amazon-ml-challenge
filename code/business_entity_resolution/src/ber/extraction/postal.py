"""Postal code extraction and comparison (Section 10.2)."""
import re
from typing import Optional, Tuple

POSTAL_PATTERNS = [
    # US ZIP+4: e.g. 94105-1234 -> 94105
    (re.compile(r"\b(\d{5})-\d{4}\b"), 1),
    # 6-digit PIN (India): e.g. 560034
    (re.compile(r"\b(\d{6})\b"), 1),
    # 5-digit ZIP / Code Postal (US, France): e.g. 94105, 75002
    (re.compile(r"\b(\d{5})\b"), 1),
    # Fallback 4-digit code
    (re.compile(r"\b(\d{4})\b"), 1),
]


def extract_postal_code(addr_raw: str, country: str = "") -> str:
    """Extracts canonical postal code from address string.
    US ZIP+4 is reduced to 5 digits.
    Unseen country: extracts the best matching postal token (preferably towards end of address).
    """
    if not addr_raw:
        return ""

    # Check US ZIP+4 first
    m_zip4 = POSTAL_PATTERNS[0][0].search(addr_raw)
    if m_zip4:
        return m_zip4.group(1)

    # Search all 5 or 6 digit candidates
    candidates = re.findall(r"\b\d{5,6}\b", addr_raw)
    if candidates:
        # Postal codes are typically towards the end of an address
        return candidates[-1]

    # Search 4-digit codes
    candidates_4 = re.findall(r"\b\d{4}\b", addr_raw)
    if candidates_4:
        return candidates_4[-1]

    return ""


def compare_postal_codes(p1: str, p2: str) -> Tuple[int, int, int]:
    """Compares two postal codes.

    Returns:
        (postal_eq, postal_both_present, postal_prefix_len)
        - postal_eq: 1 if both present and equal
        - postal_both_present: 1 if both present
        - postal_prefix_len: common prefix length (signals geographical proximity)
    """
    if not p1 or not p2:
        return 0, 0, 0

    both_present = 1
    eq = 1 if p1 == p2 else 0

    # Common prefix length
    prefix_len = 0
    for c1, c2 in zip(p1, p2):
        if c1 == c2:
            prefix_len += 1
        else:
            break

    return eq, both_present, prefix_len
