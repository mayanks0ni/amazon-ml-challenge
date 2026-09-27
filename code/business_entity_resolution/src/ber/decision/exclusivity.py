"""Exclusivity post-processing ensuring each S2/S3 record maps to at most one S1."""
from collections import defaultdict
from typing import Dict, List, Tuple


def apply_exclusivity(
    pair_probs: Dict[Tuple[str, str], float],
    delta: float = 0.20,
    shrink: float = 0.50,
    hard: bool = False,
) -> Dict[Tuple[str, str], float]:
    """Applies exclusivity resolution across candidate edges.

    Since S1 is the deduplicated reference source, each S2/S3 record should
    belong to at most one S1 entity.

    If two or more S1s compete for the same candidate record r:
    - If hard=True: only the highest-scoring S1 keeps candidate r (others set to 0.0).
    - If soft (default): if p_best - p_ir >= delta, set to 0.0; else shrink p_ir by shrink factor.

    Args:
        pair_probs: Mapping of (s1_id, rec_id) -> probability.
        delta: Margin above which competitors are zeroed out.
        shrink: Penalty factor applied to competitors when margin is small.
        hard: If True, strictly enforce 1-to-1 winner takes all.

    Returns:
        Updated pair_probs dictionary.
    """
    # Group edges by pool record (r)
    record_to_s1s = defaultdict(list)
    for (s1_id, rec_id), prob in pair_probs.items():
        if prob > 0:
            record_to_s1s[rec_id].append((s1_id, prob))

    updated_probs = dict(pair_probs)

    for rec_id, edges in record_to_s1s.items():
        if len(edges) <= 1:
            continue

        # Find best S1
        i_best, p_best = max(edges, key=lambda e: e[1])

        for s1_id, p_ir in edges:
            if s1_id != i_best:
                if hard:
                    updated_probs[(s1_id, rec_id)] = 0.0
                else:
                    if (p_best - p_ir) >= delta:
                        updated_probs[(s1_id, rec_id)] = 0.0
                    else:
                        updated_probs[(s1_id, rec_id)] = p_ir * shrink

    return updated_probs
