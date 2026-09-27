"""Channel C6: Sorted Neighbourhood Method (SNM) blocking."""
from typing import Dict, List, Sequence, Set, Tuple
from ber.io.schemas import CanonicalRecord


def retrieve_c6_snm(
    s1_records: Sequence[CanonicalRecord],
    pool_records: Sequence[CanonicalRecord],
    window: int = 20,
) -> List[Tuple[str, str, int, float]]:
    """C6: Sorted Neighbourhood Method over forward and reversed core names.
    Catches beginning and ending character typos that break hash/exact keys.
    """
    if not s1_records or not pool_records:
        return []

    results: List[Tuple[str, str, int, float]] = []
    seen_pairs: Set[Tuple[str, str]] = set()

    # Combine all records for sorting, keeping track of source
    tagged_records = [(r.name_core, r.entity_id, r.source, r.country_norm) for r in s1_records] + [
        (r.name_core, r.entity_id, r.source, r.country_norm) for r in pool_records
    ]

    # Pass 1: Forward sort
    sorted_forward = sorted(tagged_records, key=lambda x: x[0])
    n = len(sorted_forward)

    for i in range(n):
        name_i, id_i, src_i, c_i = sorted_forward[i]
        if src_i != "S1":
            continue

        # Look in window before and after
        start_idx = max(0, i - window)
        end_idx = min(n, i + window + 1)

        rank = 0
        for j in range(start_idx, end_idx):
            if i == j:
                continue
            name_j, id_j, src_j, c_j = sorted_forward[j]
            if src_j != "S1" and (id_i, id_j) not in seen_pairs:
                # Same country check if known
                if c_i and c_j and c_i != c_j:
                    continue
                seen_pairs.add((id_i, id_j))
                results.append((id_i, id_j, rank, 1.0 / (abs(i - j) + 1.0)))
                rank += 1

    # Pass 2: Reversed string sort (handles prefixes with noisy suffixes)
    sorted_rev = sorted(tagged_records, key=lambda x: x[0][::-1])
    for i in range(n):
        name_i, id_i, src_i, c_i = sorted_rev[i]
        if src_i != "S1":
            continue

        start_idx = max(0, i - window)
        end_idx = min(n, i + window + 1)

        rank = 0
        for j in range(start_idx, end_idx):
            if i == j:
                continue
            name_j, id_j, src_j, c_j = sorted_rev[j]
            if src_j != "S1" and (id_i, id_j) not in seen_pairs:
                if c_i and c_j and c_i != c_j:
                    continue
                seen_pairs.add((id_i, id_j))
                results.append((id_i, id_j, rank, 1.0 / (abs(i - j) + 1.0)))
                rank += 1

    return results
