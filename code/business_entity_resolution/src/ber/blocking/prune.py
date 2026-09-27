"""Budgeted candidate pruning to top-M while preserving multi-channel hits (Section 12.5)."""
from typing import Dict, List, Set, Tuple
from ber.io.schemas import CandidateProvenance


def count_set_bits(n: int) -> int:
    """Counts number of 1-bits in integer."""
    return bin(n).count("1")


def prune_candidates(
    candidates: Dict[str, Dict[str, CandidateProvenance]],
    budget_m: int = 60,
    keep_if_channels_ge: int = 2,
) -> Dict[str, List[str]]:
    """Prunes candidate pool to fixed budget per S1 while keeping high-confidence multi-channel hits.

    Rule from Section 11.4 & 12.5:
    - Keep every pair seen by >= keep_if_channels_ge channels (e.g. 2).
    - Fill remaining budget up to budget_m using priority rank.

    Returns:
        Dict mapping s1_id -> sorted list of candidate pool IDs (Ci).
    """
    pruned: Dict[str, List[str]] = {}

    for s1_id, pool_dict in candidates.items():
        if not pool_dict:
            pruned[s1_id] = []
            continue

        scored_candidates: List[Tuple[str, int, float]] = []

        for pool_id, prov in pool_dict.items():
            n_ch = count_set_bits(prov.mask)
            # Composite priority: higher channel count, lower min rank, higher max score
            min_rank = min(prov.rank.values()) if prov.rank else 999
            max_score = max(prov.score.values()) if prov.score else 0.0
            priority = (n_ch * 100.0) - min_rank + (max_score * 10.0)
            scored_candidates.append((pool_id, n_ch, priority))

        # Sort by priority descending
        scored_candidates.sort(key=lambda x: -x[2])

        selected: List[str] = []
        seen: Set[str] = set()

        # Step 1: Always keep multi-channel candidates
        for pool_id, n_ch, _ in scored_candidates:
            if n_ch >= keep_if_channels_ge:
                selected.append(pool_id)
                seen.add(pool_id)

        # Step 2: Fill remaining up to budget_m from top of priority list
        for pool_id, _, _ in scored_candidates:
            if len(selected) >= budget_m:
                break
            if pool_id not in seen:
                selected.append(pool_id)
                seen.add(pool_id)

        pruned[s1_id] = selected

    return pruned
