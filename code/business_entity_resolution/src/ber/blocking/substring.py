"""Channel C9: Substring / containment blocking.

Retrieves candidates where one name is a prefix or substring of the other.
Catches cases like "ABC" → "ABC International" or "Smith & Co" → "Smith & Company Ltd".
"""
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple
from ber.io.schemas import CanonicalRecord


def retrieve_c9_substring(
    s1_records: Sequence[CanonicalRecord],
    pool_records: Sequence[CanonicalRecord],
    top_k: int = 20,
    min_name_len: int = 4,
) -> List[Tuple[str, str, int, float]]:
    """C9: Substring and prefix containment retrieval.

    For each S1 record, retrieves pool records where:
    1. S1 name_core is a substring of pool name_core (or vice versa)
    2. S1 name_clean (alphanumeric-only) is a prefix of pool name_clean (or vice versa)

    Only considers names of length >= min_name_len to avoid trivial short matches.
    Additionally indexes by (prefix_5, country) to keep runtime bounded.

    Returns:
        List of (s1_id, pool_id, rank, score) tuples.
    """
    if not s1_records or not pool_records:
        return []

    # Build prefix indexes for bounded lookup
    # Key: (first 5 chars of name_clean, country) → list of (entity_id, name_core, name_clean)
    prefix5_index: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = defaultdict(list)
    # Key: (first 3 chars of name_core, country) → list of (entity_id, name_core, name_clean)
    prefix3_index: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = defaultdict(list)

    for rec in pool_records:
        if not rec.name_core or len(rec.name_core) < min_name_len:
            continue
        country = rec.country_norm or ""
        clean = rec.name_clean or ""

        if len(clean) >= 5:
            postings = prefix5_index[(clean[:5], country)]
            if len(postings) < 50:
                postings.append((rec.entity_id, rec.name_core, clean))

        if len(rec.name_core) >= 3:
            postings = prefix3_index[(rec.name_core[:3], country)]
            if len(postings) < 100:
                postings.append((rec.entity_id, rec.name_core, clean))

    results: List[Tuple[str, str, int, float]] = []

    for s1 in s1_records:
        if not s1.name_core or len(s1.name_core) < min_name_len:
            continue

        country = s1.country_norm or ""
        s1_clean = s1.name_clean or ""
        s1_core = s1.name_core

        matches: Dict[str, float] = {}
        seen: Set[str] = set()

        # Strategy 1: Prefix-5 lookup on clean name
        if len(s1_clean) >= 5:
            for pid, p_core, p_clean in prefix5_index.get((s1_clean[:5], country), []):
                if pid in seen:
                    continue
                # Check containment
                shorter = min(s1_core, p_core, key=len)
                longer = max(s1_core, p_core, key=len)
                if shorter in longer:
                    ratio = len(shorter) / len(longer)
                    if ratio >= 0.40:  # Not too short vs long
                        matches[pid] = ratio
                        seen.add(pid)

        # Strategy 2: Prefix-3 lookup on name_core
        if len(s1_core) >= 3:
            for pid, p_core, p_clean in prefix3_index.get((s1_core[:3], country), []):
                if pid in seen:
                    continue
                shorter = min(s1_core, p_core, key=len)
                longer = max(s1_core, p_core, key=len)
                if shorter in longer:
                    ratio = len(shorter) / len(longer)
                    if ratio >= 0.40:
                        matches[pid] = ratio
                        seen.add(pid)

        # Strategy 3: Check if pool name_clean starts with s1_clean or vice versa
        if len(s1_clean) >= 5:
            for pid, p_core, p_clean in prefix5_index.get((s1_clean[:5], country), []):
                if pid in seen:
                    continue
                if p_clean and (p_clean.startswith(s1_clean) or s1_clean.startswith(p_clean)):
                    shorter_len = min(len(s1_clean), len(p_clean))
                    longer_len = max(len(s1_clean), len(p_clean))
                    if longer_len > 0:
                        ratio = shorter_len / longer_len
                        if ratio >= 0.50:
                            matches[pid] = ratio
                            seen.add(pid)

        if not matches:
            continue

        sorted_matches = sorted(matches.items(), key=lambda x: -x[1])[:top_k]
        for rank, (pid, score) in enumerate(sorted_matches):
            results.append((s1.entity_id, pid, rank, score))

    return results
