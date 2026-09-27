"""Channel C8: Token-overlap inverted index blocking.

Retrieves candidates sharing multiple name tokens with S1, scored by
weighted Jaccard overlap.  Catches pairs that share 2+ tokens but where
no single specialised channel (exact / rare / phonetic) fires.
"""
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple
from ber.io.schemas import CanonicalRecord


def retrieve_c8_token_overlap(
    s1_records: Sequence[CanonicalRecord],
    pool_records: Sequence[CanonicalRecord],
    top_k: int = 30,
    min_shared: int = 2,
    max_postings: int = 500,
) -> List[Tuple[str, str, int, float]]:
    """C8: Token-overlap retrieval via full inverted index on name_core_tokens.

    Unlike C3 (rare tokens only), this uses ALL tokens but requires at least
    `min_shared` shared tokens and scores by Jaccard similarity.  Posting lists
    are capped at `max_postings` to keep runtime bounded.

    Returns:
        List of (s1_id, pool_id, rank, score) tuples.
    """
    if not s1_records or not pool_records:
        return []

    # Build inverted index on pool name_core_tokens
    token_index: Dict[str, List[str]] = defaultdict(list)
    pool_token_sets: Dict[str, Set[str]] = {}

    for rec in pool_records:
        unique_tokens = set(rec.name_core_tokens)
        pool_token_sets[rec.entity_id] = unique_tokens
        for t in unique_tokens:
            if len(t) >= 2:  # Skip single-char tokens
                postings = token_index[t]
                if len(postings) < max_postings:
                    postings.append(rec.entity_id)

    results: List[Tuple[str, str, int, float]] = []

    for s1 in s1_records:
        s1_tokens = set(s1.name_core_tokens)
        if len(s1_tokens) < 1:
            continue

        # Accumulate shared token counts per candidate
        candidate_shared: Dict[str, int] = defaultdict(int)
        for t in s1_tokens:
            if len(t) >= 2 and t in token_index:
                for pid in token_index[t]:
                    candidate_shared[pid] += 1

        if not candidate_shared:
            continue

        # Filter candidates with >= min_shared tokens (or 1 if S1 has only 1 token) and compute Jaccard
        required_shared = 1 if len(s1_tokens) == 1 else min_shared
        scored: List[Tuple[str, float]] = []
        for pid, n_shared in candidate_shared.items():
            if n_shared >= required_shared:
                pool_toks = pool_token_sets.get(pid, set())
                union_size = len(s1_tokens | pool_toks)
                if union_size > 0:
                    jaccard = n_shared / union_size
                    scored.append((pid, jaccard))

        scored.sort(key=lambda x: -x[1])
        for rank, (pid, score) in enumerate(scored[:top_k]):
            results.append((s1.entity_id, pid, rank, score))

    return results
