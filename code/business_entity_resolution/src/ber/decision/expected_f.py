"""Expected-F0.5 set selection using dynamic programming over candidate outcomes."""
import math
import numpy as np
from typing import Sequence, Tuple


def poisson_binomial(ps: Sequence[float]) -> np.ndarray:
    """Computes exact distribution of sum of independent Bernoulli variables."""
    pmf = np.array([1.0], dtype=np.float64)
    for p in ps:
        pmf = np.concatenate([pmf * (1.0 - p), [0.0]]) + np.concatenate([[0.0], pmf * p])
    return pmf


def select_expected_f05(
    p: np.ndarray,
    lam: float = 0.03,
    beta: float = 0.0,
    kmax: int = 15,
    pmin: float = 1e-6,
) -> Tuple[np.ndarray, int, float]:
    """Expected-F0.5 set selection from Appendix A reference code.

    Args:
        p: Calibrated probabilities of one S1's candidates.
        lam: Expected number of gold records missed by blocking (default estimated from ceiling).
        beta: Log-odds shift tuned by nested CV (default 0.0).
        kmax: Maximum number of candidates to select.
        pmin: Minimum probability threshold to consider.

    Returns:
        Tuple of (selected_indices, best_k, best_expected_f05).
    """
    if len(p) == 0:
        return np.array([], dtype=int), 0, 1.0

    p_arr = np.array(p, dtype=np.float64)
    # Clip probabilities to avoid division by zero or log of zero
    p_clipped = np.clip(p_arr, 1e-7, 1.0 - 1e-7)

    if beta != 0.0:
        # Log-odds shift
        logits = np.log(p_clipped / (1.0 - p_clipped)) + beta
        p_shifted = 1.0 / (1.0 + np.exp(-logits))
    else:
        p_shifted = p_clipped

    order = np.argsort(-p_shifted)
    sorted_p = p_shifted[order]

    # Filter by pmin and cap candidate length for DP efficiency (top 14)
    valid_mask = sorted_p > pmin
    valid_indices = np.where(valid_mask)[0][:14]

    if len(valid_indices) == 0:
        return np.array([], dtype=int), 0, 1.0

    ps = sorted_p[valid_indices]
    sub_order = order[valid_indices]

    # Model missed gold records U ~ Poisson(lam) truncated to 4 terms
    u = np.array([math.exp(-lam) * (lam**j) / math.factorial(j) for j in range(4)], dtype=np.float64)
    u_sum = u.sum()
    if u_sum > 0:
        u /= u_sum

    best_k = 0
    best_v = -1.0

    max_k = min(kmax, len(ps))
    for k in range(0, max_k + 1):
        pin = poisson_binomial(ps[:k])
        pout = np.convolve(poisson_binomial(ps[k:]), u)

        if k == 0:
            # Singleton credit: expected score is P(G = 0)
            v = float(pout[0])
        else:
            a = np.arange(len(pin), dtype=np.float64)[:, None]
            c = np.arange(len(pout), dtype=np.float64)[None, :]
            # Closed-form F0.5 = 5 * a / (4 * k + a + c)
            denom = 4.0 * k + a + c
            gain = np.where((a > 0) & (denom > 0), 5.0 * a / denom, 0.0)
            joint_prob = pin[:, None] * pout[None, :]
            v = float((joint_prob * gain).sum())

        if v > best_v + 1e-12:
            best_k = k
            best_v = v

    selected_indices = sub_order[:best_k]
    return selected_indices, best_k, best_v
