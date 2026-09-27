"""Decision layer: Expected-F0.5 selection, threshold rules, exclusivity, and singletons."""
from ber.decision.expected_f import select_expected_f05, poisson_binomial
from ber.decision.thresholds import (
    select_global_threshold,
    select_top1_threshold,
    select_two_thresholds,
    tune_thresholds_cv,
)
from ber.decision.exclusivity import apply_exclusivity
from ber.decision.singleton import evaluate_singletons

__all__ = [
    "select_expected_f05",
    "poisson_binomial",
    "select_global_threshold",
    "select_top1_threshold",
    "select_two_thresholds",
    "tune_thresholds_cv",
    "apply_exclusivity",
    "evaluate_singletons",
]
