"""Blocking evaluation metrics (Section 11.3)."""
from typing import Dict, List, Sequence, Set
import numpy as np
from ber.evaluation.f05 import compute_entity_f05


def compute_blocking_metrics(
    candidates: Dict[str, Sequence[str]],
    ground_truth: Dict[str, Sequence[str]],
    num_s1: int,
    num_pool: int,
) -> Dict[str, float]:
    """Computes standard blocking quality metrics:
    - Pair Completeness (PC): fraction of all ground truth pairs retrieved.
    - Reduction Ratio (RR): fraction of Cartesian product pruned away.
    - Entity Recall (ER): fraction of S1 entities where ALL gold matches are retrieved.
    - Ceiling F0.5: upper bound on achievable F0.5 given the retrieved candidate sets.
    """
    total_gold_pairs = 0
    captured_gold_pairs = 0
    total_candidates = 0

    entity_recalls = []
    ceiling_scores = []
    candidate_counts = []

    for s1_id, gold_list in ground_truth.items():
        cand_list = candidates.get(s1_id, [])
        cand_set = set(cand_list)
        gold_set = set(gold_list)

        candidate_counts.append(len(cand_list))
        total_candidates += len(cand_list)
        total_gold_pairs += len(gold_set)

        if len(gold_set) > 0:
            intersection = gold_set.intersection(cand_set)
            captured_gold_pairs += len(intersection)
            # Entity Recall: 1 if all gold are in candidate set
            entity_recalls.append(1.0 if len(intersection) == len(gold_set) else 0.0)
            # Ceiling: F0.5 score of oracle matcher predicting G_i \cap C_i
            oracle_pred = list(intersection)
            ceiling_scores.append(compute_entity_f05(oracle_pred, gold_list))
        else:
            # Singleton entity
            entity_recalls.append(1.0)
            ceiling_scores.append(1.0)

    pc = (captured_gold_pairs / total_gold_pairs) if total_gold_pairs > 0 else 1.0
    cartesian_space = float(num_s1 * num_pool)
    rr = 1.0 - (total_candidates / cartesian_space) if cartesian_space > 0 else 1.0
    er = float(np.mean(entity_recalls)) if entity_recalls else 1.0
    ceiling = float(np.mean(ceiling_scores)) if ceiling_scores else 1.0

    counts_arr = np.array(candidate_counts) if candidate_counts else np.array([0])

    return {
        "pair_completeness": pc,
        "reduction_ratio": rr,
        "entity_recall": er,
        "ceiling_f05": ceiling,
        "mean_candidates_per_s1": float(np.mean(counts_arr)),
        "p50_candidates_per_s1": float(np.percentile(counts_arr, 50)),
        "p99_candidates_per_s1": float(np.percentile(counts_arr, 99)),
        "max_candidates_per_s1": float(np.max(counts_arr)),
    }
