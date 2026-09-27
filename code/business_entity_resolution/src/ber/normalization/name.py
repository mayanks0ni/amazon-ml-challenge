"""Business-name normalization pipeline (Section 10.1)."""
import re
from typing import List, Tuple
from ber.normalization.unicode import normalize_unicode
from ber.normalization.fold import fold_text
from ber.extraction.legal_form import extract_legal_form
from ber.extraction.dba import extract_dba_aliases

# Generic stop words to omit from name_core
CORE_NAME_STOP_WORDS = {
    "the", "and", "of", "for", "in", "at", "by", "to", "a", "an", "co", "company"
}


def collapse_repeated_tokens(tokens: List[str]) -> List[str]:
    """Collapses consecutive repeated tokens: e.g. ['new', 'new', 'delhi'] -> ['new', 'delhi']."""
    if not tokens:
        return []
    res = [tokens[0]]
    for t in tokens[1:]:
        if t != res[-1]:
            res.append(t)
    return res


def extract_acronym(tokens: List[str]) -> str:
    """Computes acronym from initials of core tokens if >= 2 tokens."""
    if len(tokens) < 2:
        return ""
    return "".join(t[0] for t in tokens if t and t[0].isalnum())


def normalize_business_name(raw_name: str) -> dict:
    """Executes the full business-name normalization pipeline (Section 10.1).

    Returns a dictionary of normalized representations:
        - name_norm
        - name_core
        - name_alts
        - legal_form
        - name_fold
        - name_acronym
        - name_tokens
        - name_core_tokens
        - name_missing
    """
    if not raw_name or not raw_name.strip():
        return {
            "name_norm": "",
            "name_core": "",
            "name_alts": [],
            "legal_form": "",
            "name_fold": "",
            "name_acronym": "",
            "name_tokens": [],
            "name_core_tokens": [],
            "name_missing": True,
        }

    # Step 1: Unicode NFKC + casefold
    norm = normalize_unicode(raw_name)

    # Step 2: &, +, @ -> and, at
    norm = norm.replace("&", " and ").replace("+", " and ").replace("@", " at ")

    # Step 3: Extract DBA / parenthetical aliases
    primary_name, raw_alts = extract_dba_aliases(norm)

    # Step 3b: Strip web URLs and domain extensions (e.g., .com, .in, .org, www.)
    name_no_url = re.sub(r"https?://", "", primary_name)
    name_no_url = re.sub(r"\bwww\.", "", name_no_url)
    domain_stripped = re.sub(
        r"\.(?:com|org|net|co\.in|in|co|io|biz|info|gov|edu|fr|us)(?=[/\s]|$)",
        "",
        name_no_url,
    ).strip()
    if domain_stripped and domain_stripped != primary_name:
        raw_alts.append(domain_stripped)
        primary_name = domain_stripped

    # Step 4: Extract legal form from primary name
    core, legal_form = extract_legal_form(primary_name)

    # Normalize punctuation while keeping joined tokens for hyphen/dot
    # e.g., 7-eleven -> '7 eleven'
    norm_clean = re.sub(r"[^a-z0-9]+", " ", norm).strip()
    core_clean = re.sub(r"[^a-z0-9]+", " ", core).strip()

    name_tokens = [t for t in norm_clean.split() if t]
    name_tokens = collapse_repeated_tokens(name_tokens)

    core_tokens_raw = [t for t in core_clean.split() if t]
    core_tokens_raw = collapse_repeated_tokens(core_tokens_raw)
    core_tokens = [t for t in core_tokens_raw if t not in CORE_NAME_STOP_WORDS]
    if not core_tokens:
        core_tokens = core_tokens_raw  # Fallback if all were stop words

    final_core = " ".join(core_tokens)
    acronym = extract_acronym(core_tokens)
    folded = fold_text(final_core)
    clean_name = re.sub(r"[^a-z0-9]", "", final_core)

    # Process alternate names
    name_alts = []
    for alt in raw_alts:
        alt_norm = re.sub(r"[^a-z0-9]+", " ", normalize_unicode(alt)).strip()
        if alt_norm and alt_norm != norm_clean:
            name_alts.append(alt_norm)

    return {
        "name_norm": " ".join(name_tokens),
        "name_core": final_core,
        "name_clean": clean_name,
        "name_alts": name_alts,
        "legal_form": legal_form,
        "name_fold": folded,
        "name_acronym": acronym,
        "name_tokens": name_tokens,
        "name_core_tokens": core_tokens,
        "name_missing": False,
    }
