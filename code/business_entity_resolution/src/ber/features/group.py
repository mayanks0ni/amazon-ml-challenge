"""Group-context feature generation for Stage 2b LightGBM (Section 13.4)."""
from collections import defaultdict
import math
from typing import Dict, List, Sequence, Tuple
import numpy as np


def compute_group_context_features(
    pair_keys: Sequence[Tuple[str, str]],
    probabilities: Sequence[float],
) -> List[Dict[str, float]]:
    """Computes Stage 2b collective group-context features from Stage 2a probabilities.

    Features capture:
    - Within-S1 competition ('is this the best candidate?')
    - Reverse competition ('is this candidate contested by another S1?')
    - S1 uncertainty and multi-match density
    """
    s1_to_candidates: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    pool_to_s1s: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    for (s1_id, pool_id), p in zip(pair_keys, probabilities):
        s1_to_candidates[s1_id].append((pool_id, float(p)))
        pool_to_s1s[pool_id].append((s1_id, float(p)))

    # Precompute S1-level statistics
    s1_stats: Dict[str, dict] = {}
    for s1_id, cands in s1_to_candidates.items():
        sorted_cands = sorted(cands, key=lambda x: -x[1])
        probs = np.array([x[1] for x in sorted_cands], dtype=np.float64)
        p_best = float(probs[0]) if len(probs) > 0 else 0.0
        p_second = float(probs[1]) if len(probs) > 1 else 0.0
        n_above_half = float(np.sum(probs >= 0.50))
        p_sum = float(np.sum(probs))

        # Entropy of candidate probabilities
        if p_sum > 0:
            norm_p = probs / p_sum
            entropy = float(-np.sum(norm_p * np.log(norm_p + 1e-12)))
        else:
            entropy = 0.0

        rank_map = {item[0]: rank for rank, item in enumerate(sorted_cands)}

        s1_stats[s1_id] = {
            "p_best": p_best,
            "p_second": p_second,
            "n_above_half": n_above_half,
            "p_sum": p_sum,
            "entropy": entropy,
            "rank_map": rank_map,
        }

    # Precompute Pool-level statistics (Reverse rank)
    pool_stats: Dict[str, dict] = {}
    for pool_id, s1s in pool_to_s1s.items():
        sorted_s1s = sorted(s1s, key=lambda x: -x[1])
        rev_rank_map = {item[0]: rank for rank, item in enumerate(sorted_s1s)}
        top_p = sorted_s1s[0][1] if len(sorted_s1s) > 0 else 0.0
        second_p = sorted_s1s[1][1] if len(sorted_s1s) > 1 else 0.0
        pool_stats[pool_id] = {
            "rev_rank_map": rev_rank_map,
            "top_p": top_p,
            "second_p": second_p,
        }

    # Generate feature dictionaries for each pair
    features_list: List[Dict[str, float]] = []

    for (s1_id, pool_id), p in zip(pair_keys, probabilities):
        st = s1_stats[s1_id]
        pst = pool_stats[pool_id]

        p_rank = float(st["rank_map"].get(pool_id, 999))
        p_gap_to_best = float(st["p_best"] - p)

        rev_rank = float(pst["rev_rank_map"].get(s1_id, 999))
        # Reverse gap: if s1 is top, gap to second; else gap to top (negative)
        if rev_rank == 0:
            rev_gap = float(p - pst["second_p"])
        else:
            rev_gap = float(p - pst["top_p"])

        feats = {
            "p_stage2a": float(p),
            "p_rank_in_s1": p_rank,
            "p_gap_to_best": p_gap_to_best,
            "p_best_in_s1": st["p_best"],
            "p_second_in_s1": st["p_second"],
            "n_above_0.5_in_s1": st["n_above_half"],
            "p_sum_in_s1": st["p_sum"],
            "p_entropy_in_s1": st["entropy"],
            "rev_rank": rev_rank,
            "rev_gap": rev_gap,
        }
        features_list.append(feats)

    return features_list
