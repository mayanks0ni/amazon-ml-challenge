"""Unit tests for official F0.5 metric computation."""
import math
import pytest
from ber.evaluation.f05 import compute_entity_f05, compute_macro_f05


def test_f05_worked_example():
    # Official worked example from problem statement reproduces 0.714:
    # Pred: S1-00001 matches [S2-00047, S2-00193, S3-00812]
    # Gold: S1-00001 matches [S2-00047, S3-00812]
    # Precision = 2/3, Recall = 2/2 = 1.0
    # F0.5 = (1.25 * 0.667 * 1.0) / (0.25 * 0.667 + 1.0) = 0.714
    pred = ["S2-00047", "S2-00193", "S3-00812"]
    gold = ["S2-00047", "S3-00812"]
    score = compute_entity_f05(pred, gold)
    assert round(score, 3) == 0.714
    assert math.isclose(score, 5.0 / 7.0, rel_tol=1e-9)


def test_f05_formula_equivalence():
    # Compare closed form against traditional precision/recall formula
    for tp, n_pred, n_gold in [
        (1, 1, 1),
        (1, 2, 2),
        (2, 2, 3),
        (2, 4, 3),
        (3, 5, 4),
    ]:
        p = tp / n_pred
        r = tp / n_gold
        expected_official = 1.25 * p * r / (0.25 * p + r)
        closed_form = (5.0 * tp) / (4.0 * n_pred + n_gold)
        assert math.isclose(expected_official, closed_form, rel_tol=1e-12)


def test_f05_singletons():
    # Singleton correctly predicted empty -> 1.0
    assert compute_entity_f05([], []) == 1.0
    # Singleton incorrectly predicted with candidates -> 0.0
    assert compute_entity_f05(["S2-001"], []) == 0.0
    # Non-singleton missed completely (predicted empty) -> 0.0
    assert compute_entity_f05([], ["S2-001"]) == 0.0


def test_macro_averaging():
    gt = {
        "S1-1": [],  # singleton
        "S1-2": ["S2-1"],  # 1 gold
        "S1-3": ["S2-2", "S3-3"],  # 2 gold
    }
    preds = {
        "S1-1": [],  # correct singleton: 1.0
        "S1-2": ["S2-1"],  # 1/1 correct: 5*1/(4*1+1) = 5/5 = 1.0
        "S1-3": ["S2-2"],  # 1 correct out of 1 pred, 2 gold: 5*1/(4*1+2) = 5/6 = 0.8333...
    }
    res = compute_macro_f05(preds, gt)
    expected_macro = (1.0 + 1.0 + (5.0 / 6.0)) / 3.0
    assert math.isclose(res["macro_f05"], expected_macro, rel_tol=1e-6)
    assert res["singleton_acc"] == 1.0
    assert res["num_singletons"] == 1.0
    assert res["num_entities"] == 3.0
