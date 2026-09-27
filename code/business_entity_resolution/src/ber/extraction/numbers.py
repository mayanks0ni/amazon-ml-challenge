"""Numeric token and house number extraction (Section 10.2)."""
import re
from typing import List, Optional, Set, Tuple

HOUSE_NO_RE = re.compile(r"^\s*(?:plot\s+no\.?\s*)?([0-9]+[a-z]?|[0-9]+/[0-9]+)\b", re.IGNORECASE)
UNIT_RE = re.compile(
    r"\b(?:suite|ste|shop|unit|flat|fl|floor|apt|apartment|room|rm)\s*([0-9a-z\-/]+)\b",
    re.IGNORECASE,
)
NUM_TOKEN_RE = re.compile(r"\b([0-9]+(?:[/\-][0-9a-z]+)?|[0-9]+[a-z]?)\b")


def extract_address_numbers(addr_norm: str) -> Tuple[List[str], str, str]:
    """Extracts all digit-bearing tokens, leading house/street number, and unit/shop number.

    Returns:
        (addr_nums, house_no, unit)
    """
    if not addr_norm:
        return [], "", ""

    # All numbers / compound numbers
    matches = NUM_TOKEN_RE.findall(addr_norm)
    addr_nums = [m.lower().strip() for m in matches if any(c.isdigit() for c in m)]

    # Leading house/street number
    house_match = HOUSE_NO_RE.match(addr_norm)
    house_no = house_match.group(1).lower() if house_match else ""

    # Unit / suite / shop number
    unit_match = UNIT_RE.search(addr_norm)
    unit = unit_match.group(1).lower() if unit_match else ""

    return addr_nums, house_no, unit


def compare_house_numbers(h1: str, h2: str) -> Tuple[int, int]:
    """Compares house numbers between two addresses.

    Returns:
        (house_eq, house_conflict)
        - house_eq: 1 if both present and equal, 0 otherwise
        - house_conflict: 1 if both present and different (strong branch evidence!), 0 otherwise
    """
    if not h1 or not h2:
        return 0, 0
    if h1 == h2:
        return 1, 0
    return 0, 1


def compare_unit_numbers(u1: str, u2: str) -> Tuple[int, int]:
    """Compares unit numbers between two addresses.

    Returns:
        (unit_eq, unit_conflict)
    """
    if not u1 or not u2:
        return 0, 0
    if u1 == u2:
        return 1, 0
    return 0, 1
