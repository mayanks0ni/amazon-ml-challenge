"""Paired bootstrap significance testing over S1 entities (Section 21.3)."""
from typing import Dict, List, Sequence, Tuple
import numpy as np
from ber.evaluation.f05 import compute_entity_f05


def paired_bootstrap_f05_diff(
    preds_model_a: Dict[str, Sequence[str]],
    preds_model_b: Dict[str, Sequence[str]],
    ground_truth: Dict[str, Sequence[str]],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Computes paired bootstrap confidence interval for delta = F0.5(B) - F0.5(A).

    Returns:
        dict with: 'mean_delta', 'ci_lower', 'ci_upper', 'p_value'
    """
    s1_ids = list(ground_truth.keys())
    n = len(s1_ids)
    if n == 0:
        return {"mean_delta": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0}

    # Compute per-entity scores for both models
    scores_a = np.array([compute_entity_f05(preds_model_a.get(sid, []), ground_truth[sid]) for sid in s1_ids])
    scores_b = np.array([compute_entity_f05(preds_model_b.get(sid, []), ground_truth[sid]) for sid in s1_ids])

    deltas = scores_b - scores_a
    observed_mean_delta = float(np.mean(deltas))

    rng = np.random.default_rng(seed)
    bootstrap_deltas = np.zeros(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        sample_indices = rng.integers(0, n, size=n)
        bootstrap_deltas[i] = np.mean(deltas[sample_indices])

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(bootstrap_deltas, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_deltas, 100.0 * (1.0 - alpha / 2.0)))

    # Empirical p-value that delta <= 0
    p_value = float(np.mean(bootstrap_deltas <= 0.0))

    return {
        "observed_mean_delta": observed_mean_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "is_significant": ci_lower > 0.0,
    }
