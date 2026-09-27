"""Report card generation for validation experiments (Section 21.3)."""
import json
from typing import Dict, List, Sequence
from ber.evaluation.f05 import compute_macro_f05
from ber.decision.singleton import evaluate_singletons


def generate_experiment_report(
    predictions: Dict[str, Sequence[str]],
    candidates: Dict[str, Sequence[str]],
    ground_truth: Dict[str, Sequence[str]],
    blocking_metrics: Dict[str, float],
    calibration_metrics: Dict[str, float],
    run_name: str = "run",
) -> Dict[str, any]:
    """Generates a comprehensive report card covering all Section 21.3 metrics."""
    f05_res = compute_macro_f05(predictions, ground_truth)
    singleton_res = evaluate_singletons(predictions, ground_truth)

    report = {
        "run_name": run_name,
        "overall_macro_f05": f05_res["macro_f05"],
        "non_singleton_f05": f05_res["non_singleton_f05"],
        "singleton_accuracy": singleton_res["singleton_accuracy"],
        "singleton_false_positive_rate": singleton_res["singleton_false_positive_rate"],
        "non_singleton_abstention_rate": singleton_res["non_singleton_abstention_rate"],
        # Blocking metrics
        "pair_completeness": blocking_metrics.get("pair_completeness", 0.0),
        "blocking_ceiling_f05": blocking_metrics.get("ceiling_f05", 0.0),
        "blocking_reduction_ratio": blocking_metrics.get("reduction_ratio", 0.0),
        # Calibration metrics
        "raw_ece": calibration_metrics.get("raw_ece", 0.0),
        "calibrated_ece": calibration_metrics.get("calibrated_ece", 0.0),
        "calibrated_brier": calibration_metrics.get("calibrated_brier", 0.0),
        "calibrated_logloss": calibration_metrics.get("calibrated_logloss", 0.0),
    }

    return report
