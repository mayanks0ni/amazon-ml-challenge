"""Singleton detection and evaluation module."""
from typing import Dict, List, Sequence, Set


def evaluate_singletons(
    predictions: Dict[str, Sequence[str]],
    ground_truth: Dict[str, Sequence[str]],
) -> Dict[str, float]:
    """Calculates singleton-specific performance metrics.

    - Singleton accuracy: share of gold-empty S1s predicted empty (1 - false positive rate).
    - Non-singleton abstention rate: share of gold-non-empty S1s predicted empty.
    """
    gold_empty_total = 0
    gold_empty_correct = 0
    gold_nonempty_total = 0
    gold_nonempty_abstained = 0

    for sid, gold in ground_truth.items():
        pred = predictions.get(sid, [])
        is_gold_empty = len(gold) == 0
        is_pred_empty = len(pred) == 0

        if is_gold_empty:
            gold_empty_total += 1
            if is_pred_empty:
                gold_empty_correct += 1
        else:
            gold_nonempty_total += 1
            if is_pred_empty:
                gold_nonempty_abstained += 1

    singleton_acc = (gold_empty_correct / gold_empty_total) if gold_empty_total > 0 else 0.0
    singleton_fpr = 1.0 - singleton_acc
    abstention_rate = (gold_nonempty_abstained / gold_nonempty_total) if gold_nonempty_total > 0 else 0.0

    return {
        "singleton_accuracy": singleton_acc,
        "singleton_false_positive_rate": singleton_fpr,
        "non_singleton_abstention_rate": abstention_rate,
        "total_gold_singletons": float(gold_empty_total),
        "total_gold_non_singletons": float(gold_nonempty_total),
    }
