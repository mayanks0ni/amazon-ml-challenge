"""Official F0.5 evaluation metric implementation and closed form."""
from typing import Collection, Dict, List, Sequence, Set, Union


def compute_entity_f05(pred: Collection[str], gold: Collection[str]) -> float:
    """Computes F0.5 for a single S1 entity according to the official definition.

    Closed form:
        F0.5(Pred, Gold) = 5 * TP / (4 * |Pred| + |Gold|)

    Special singleton cases:
        - If gold is empty and pred is empty -> 1.0 (singleton reward)
        - If gold is empty and pred is non-empty -> 0.0
        - If gold is non-empty and pred is empty -> 0.0
    """
    pred_set: Set[str] = set(pred)
    gold_set: Set[str] = set(gold)

    n_pred = len(pred_set)
    n_gold = len(gold_set)

    if n_gold == 0:
        return 1.0 if n_pred == 0 else 0.0

    if n_pred == 0:
        return 0.0

    tp = len(pred_set.intersection(gold_set))
    if tp == 0:
        return 0.0

    # Official closed form: 5 * TP / (4 * |Pred| + |Gold|)
    # Equivalent to 1.25 * P * R / (0.25 * P + R)
    return float((5.0 * tp) / (4.0 * n_pred + n_gold))


def compute_macro_f05(
    predictions: Dict[str, Sequence[str]],
    ground_truth: Dict[str, Sequence[str]],
) -> Dict[str, float]:
    """Computes macro-averaged per-entity F0.5 across all S1 entities in ground truth.

    Returns:
        Dictionary with:
            - 'macro_f05': Macro-averaged F0.5
            - 'singleton_acc': Singleton accuracy (share of gold-empty S1 predicted empty)
            - 'non_singleton_f05': Average F0.5 on non-singletons
            - 'num_entities': Total S1 entities evaluated
            - 'num_singletons': Total gold singletons
    """
    scores: List[float] = []
    singleton_correct: int = 0
    singleton_total: int = 0
    non_singleton_scores: List[float] = []

    for s1_id, gold_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, [])
        score = compute_entity_f05(pred_ids, gold_ids)
        scores.append(score)

        if len(gold_ids) == 0:
            singleton_total += 1
            if len(pred_ids) == 0:
                singleton_correct += 1
        else:
            non_singleton_scores.append(score)

    macro_f05 = float(sum(scores) / len(scores)) if scores else 0.0
    singleton_acc = float(singleton_correct / singleton_total) if singleton_total > 0 else 0.0
    non_singleton_f05 = (
        float(sum(non_singleton_scores) / len(non_singleton_scores))
        if non_singleton_scores
        else 0.0
    )

    return {
        "macro_f05": macro_f05,
        "singleton_acc": singleton_acc,
        "non_singleton_f05": non_singleton_f05,
        "num_entities": float(len(scores)),
        "num_singletons": float(singleton_total),
    }
