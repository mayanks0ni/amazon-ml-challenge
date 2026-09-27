"""Channel C1: Exact and structured key blocking."""
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple
from ber.io.schemas import CanonicalRecord


def retrieve_c1_exact(
    s1_records: Sequence[CanonicalRecord],
    pool_records: Sequence[CanonicalRecord],
    cap_per_s1: int = 50,
) -> List[Tuple[str, str, int, float]]:
    """C1: Exact key matches.
    Keys:
    - (name_core, country_norm)
    - (name_core, postal) [when postal present]
    - (name_fold, country_norm)

    Returns:
        List of (s1_id, pool_id, rank, score) tuples.
    """
    index_core_country: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    index_core_postal: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    index_fold_country: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    index_clean_country: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for rec in pool_records:
        if rec.name_core:
            if rec.country_norm:
                index_core_country[(rec.name_core, rec.country_norm)].append(rec.entity_id)
            if rec.postal:
                index_core_postal[(rec.name_core, rec.postal)].append(rec.entity_id)
        if rec.name_fold and rec.country_norm:
            index_fold_country[(rec.name_fold, rec.country_norm)].append(rec.entity_id)
        if rec.name_clean and rec.country_norm:
            plist = index_clean_country[(rec.name_clean, rec.country_norm)]
            if len(plist) < 50:
                plist.append(rec.entity_id)
        for alt in rec.name_alts:
            if alt and rec.country_norm:
                alt_clean = "".join(ch for ch in alt.lower() if ch.isalnum())
                if alt_clean:
                    plist = index_clean_country[(alt_clean, rec.country_norm)]
                    if len(plist) < 50:
                        plist.append(rec.entity_id)

    results: List[Tuple[str, str, int, float]] = []

    for s1 in s1_records:
        matched_pool_ids: List[str] = []
        seen: Set[str] = set()

        # 1. name_core + country
        if s1.name_core and s1.country_norm:
            for pid in index_core_country.get((s1.name_core, s1.country_norm), []):
                if pid not in seen:
                    seen.add(pid)
                    matched_pool_ids.append(pid)

        # 2. name_core + postal
        if s1.name_core and s1.postal:
            for pid in index_core_postal.get((s1.name_core, s1.postal), []):
                if pid not in seen:
                    seen.add(pid)
                    matched_pool_ids.append(pid)

        # 3. name_fold + country
        if s1.name_fold and s1.country_norm:
            for pid in index_fold_country.get((s1.name_fold, s1.country_norm), []):
                if pid not in seen:
                    seen.add(pid)
                    matched_pool_ids.append(pid)

        # 4. clean name + country
        if s1.name_clean and s1.country_norm:
            for pid in index_clean_country.get((s1.name_clean, s1.country_norm), []):
                if pid not in seen:
                    seen.add(pid)
                    matched_pool_ids.append(pid)

        # Truncate to cap
        for rank, pid in enumerate(matched_pool_ids[:cap_per_s1]):
            results.append((s1.entity_id, pid, rank, 1.0))

    return results
