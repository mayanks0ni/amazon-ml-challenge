"""Unit tests for decision rules, set selection, and exclusivity."""
import numpy as np
import pytest
from ber.decision.expected_f import select_expected_f05
from ber.decision.thresholds import select_two_thresholds, select_global_threshold
from ber.decision.exclusivity import apply_exclusivity


def test_expected_f05_singleton():
    # Very weak candidates -> should predict k = 0 (empty singleton)
    weak_probs = np.array([0.05, 0.02, 0.01])
    indices, k, v = select_expected_f05(weak_probs, lam=0.03)
    assert k == 0
    assert len(indices) == 0


def test_expected_f05_strong_candidates():
    # Strong single candidate -> should predict k = 1
    probs = np.array([0.92, 0.10])
    indices, k, v = select_expected_f05(probs, lam=0.03)
    assert k == 1
    assert len(indices) == 1
    assert indices[0] == 0


def test_two_threshold_rule():
    # Candidate 1 has 0.55 (>= t_first 0.50), Candidate 2 has 0.60 (< t_extra 0.70)
    # Only Candidate 1 should be selected
    probs = [0.55, 0.60]
    # Note: sorted order is [0.60, 0.55]
    # First is 0.60 (>= 0.50), second is 0.55 (< 0.70)
    sel = select_two_thresholds(probs, t_first=0.50, t_extra=0.70)
    assert len(sel) == 1
    assert sel[0] == 1  # 0.60 was at index 1


def test_exclusivity():
    # Two S1s competing for record S2-99: S1-A (p=0.85) vs S1-B (p=0.40)
    pair_probs = {
        ("S1-A", "S2-99"): 0.85,
        ("S1-B", "S2-99"): 0.40,
    }
    updated = apply_exclusivity(pair_probs, delta=0.20, hard=False)
    # The gap is 0.45 >= 0.20, so S1-B should be zeroed out
    assert updated[("S1-A", "S2-99")] == 0.85
    assert updated[("S1-B", "S2-99")] == 0.0
