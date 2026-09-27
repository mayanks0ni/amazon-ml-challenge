"""Address normalization pipeline (Section 10.2)."""
import re
from typing import Dict, List, Tuple
from ber.normalization.unicode import normalize_unicode
from ber.normalization.fold import fold_text
from ber.extraction.numbers import extract_address_numbers
from ber.extraction.postal import extract_postal_code
from ber.extraction.landmark import extract_landmark_phrase

# Common street type abbreviations
STREET_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "hwy": "highway",
    "pkwy": "parkway",
    "str": "street",
    "marg": "road",
    "gali": "street",
    "galli": "street",
    "nagar": "nagar",
    "ste": "suite",
    "apt": "apartment",
    "flr": "floor",
}


def expand_street_abbreviations(text: str) -> str:
    """Expands standard street abbreviations when matched as full tokens."""
    tokens = text.split()
    expanded = [STREET_ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(expanded)


def normalize_address(raw_addr: str, country: str = "") -> dict:
    """Executes the full address normalization pipeline (Section 10.2).

    Returns a dictionary of normalized representations:
        - addr_norm
        - addr_tokens
        - addr_fold
        - addr_nums
        - postal
        - house_no
        - unit
        - landmark
        - addr_tail
        - addr_missing
        - postal_missing
        - house_no_missing
    """
    if not raw_addr or not raw_addr.strip():
        return {
            "addr_norm": "",
            "addr_tokens": [],
            "addr_fold": "",
            "addr_nums": [],
            "postal": "",
            "house_no": "",
            "unit": "",
            "landmark": "",
            "addr_tail": "",
            "addr_missing": True,
            "postal_missing": True,
            "house_no_missing": True,
        }

    # Step 1: Unicode NFKC + casefold
    norm = normalize_unicode(raw_addr)

    # Step 2: Extract tail segments before stripping punctuation (city/state proxy)
    comma_segments = [s.strip() for s in norm.split(",") if s.strip()]
    addr_tail = " ".join(comma_segments[-2:]) if len(comma_segments) >= 2 else (comma_segments[0] if comma_segments else "")
    addr_tail_clean = re.sub(r"[^a-z0-9]+", " ", addr_tail).strip()

    # Step 3: Extract landmark phrase
    cleaned_addr, landmark = extract_landmark_phrase(norm)

    # Step 4: Extract postal code
    postal = extract_postal_code(raw_addr, country=country)

    # Step 5: Extract numbers, house number, and unit
    addr_nums, house_no, unit = extract_address_numbers(cleaned_addr)

    # Step 6: Expand street abbreviations
    expanded = expand_street_abbreviations(cleaned_addr)

    # Step 7: Clean punctuation
    norm_clean = re.sub(r"[^a-z0-9/]+", " ", expanded).strip()
    addr_tokens = [t for t in norm_clean.split() if t]

    # Step 8: Transliteration folding for addresses
    addr_fold = fold_text(" ".join(addr_tokens))

    # Step 9: Clean compact and sorted keys for cross-lingual / typo matching
    addr_clean = re.sub(r"[^a-z0-9]", "", " ".join(addr_tokens))

    s_ord = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", norm_clean.lower())
    s_toks = re.findall(r"[a-z0-9]+", s_ord)
    addr_stop = {
        "st", "street", "ave", "avenue", "rd", "road", "blvd", "dr", "drive",
        "ln", "lane", "ct", "court", "pl", "place", "terrace", "null", "township",
        "apt", "suite", "ste", "flr", "floor"
    }
    meaningful = sorted([t for t in s_toks if t not in addr_stop])
    sorted_addr = "".join(meaningful[:5])

    return {
        "addr_norm": " ".join(addr_tokens),
        "addr_tokens": addr_tokens,
        "addr_fold": addr_fold,
        "addr_nums": addr_nums,
        "addr_clean": addr_clean,
        "sorted_addr": sorted_addr,
        "postal": postal,
        "house_no": house_no,
        "unit": unit,
        "landmark": landmark,
        "addr_tail": addr_tail_clean,
        "addr_missing": False,
        "postal_missing": len(postal) == 0,
        "house_no_missing": len(house_no) == 0,
    }
