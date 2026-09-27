"""Challenger decision threshold rules (R2, R3, R4) and parameter tuning."""
import numpy as np
from typing import Dict, List, Sequence, Tuple
from ber.evaluation.f05 import compute_macro_f05


def select_global_threshold(p: Sequence[float], threshold: float = 0.50) -> np.ndarray:
    """R2: Global threshold rule. Accepts all candidates with p >= threshold."""
    p_arr = np.array(p, dtype=np.float64)
    return np.where(p_arr >= threshold)[0]


def select_top1_threshold(p: Sequence[float], threshold: float = 0.50) -> np.ndarray:
    """R3: Top-1 only threshold rule. Accepts the top candidate if p >= threshold."""
    if len(p) == 0:
        return np.array([], dtype=int)
    p_arr = np.array(p, dtype=np.float64)
    top_idx = int(np.argmax(p_arr))
    if p_arr[top_idx] >= threshold:
        return np.array([top_idx], dtype=int)
    return np.array([], dtype=int)


def select_two_thresholds(
    p: Sequence[float],
    t_first: float = 0.50,
    t_extra: float = 0.67,
) -> np.ndarray:
    """R4: Pre-registered challenger. Two-threshold rule.
    First match requires p >= t_first (reflecting 0.50 break-even vs empty).
    Additional matches require p >= t_extra (reflecting >= 0.667 break-even).
    """
    if len(p) == 0:
        return np.array([], dtype=int)
    p_arr = np.array(p, dtype=np.float64)
    order = np.argsort(-p_arr)

    selected: List[int] = []
    for rank, idx in enumerate(order):
        prob = p_arr[idx]
        thresh = t_first if rank == 0 else t_extra
        if prob >= thresh:
            selected.append(int(idx))
        else:
            break
    return np.array(selected, dtype=int)


def tune_thresholds_cv(
    entity_candidates: Dict[str, List[str]],
    entity_probs: Dict[str, List[float]],
    ground_truth: Dict[str, List[str]],
    rule: str = "two_threshold",
) -> Tuple[Dict[str, float], float]:
    """Sweeps thresholds over validation folds to find optimal parameters."""
    best_params: Dict[str, float] = {}
    best_score = -1.0

    if rule == "global":
        for t in np.linspace(0.2, 0.85, 27):
            preds = {}
            for sid, cands in entity_candidates.items():
                probs = entity_probs.get(sid, [])
                sel_idx = select_global_threshold(probs, threshold=float(t))
                preds[sid] = [cands[i] for i in sel_idx]
            res = compute_macro_f05(preds, ground_truth)
            if res["macro_f05"] > best_score:
                best_score = res["macro_f05"]
                best_params = {"threshold": float(t)}

    elif rule == "two_threshold":
        for t1 in np.linspace(0.35, 0.65, 7):
            for t2 in np.linspace(0.55, 0.85, 7):
                if t2 < t1:
                    continue
                preds = {}
                for sid, cands in entity_candidates.items():
                    probs = entity_probs.get(sid, [])
                    sel_idx = select_two_thresholds(probs, t_first=float(t1), t_extra=float(t2))
                    preds[sid] = [cands[i] for i in sel_idx]
                res = compute_macro_f05(preds, ground_truth)
                if res["macro_f05"] > best_score:
                    best_score = res["macro_f05"]
                    best_params = {"t_first": float(t1), "t_extra": float(t2)}

    return best_params, best_score
