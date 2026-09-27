"""Probability calibration using Isotonic Regression (Section 17)."""
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across n_bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        if i == n_bins - 1:
            idx = np.where((y_prob >= bin_lower) & (y_prob <= bin_upper))[0]
        else:
            idx = np.where((y_prob >= bin_lower) & (y_prob < bin_upper))[0]

        if len(idx) > 0:
            prop_true = np.mean(y_true[idx])
            avg_prob = np.mean(y_prob[idx])
            ece += (len(idx) / total_samples) * abs(prop_true - avg_prob)

    return float(ece)


class CrossFittedIsotonicCalibrator:
    """Isotonic regression calibrator with cross-fitting support and metric tracking."""

    def __init__(self):
        self.calibrators: List[IsotonicRegression] = []
        self.global_calibrator: Optional[IsotonicRegression] = None

    def fit_global(self, raw_probs: np.ndarray, y_true: np.ndarray) -> None:
        """Fits a single calibrator over all out-of-fold probabilities."""
        self.global_calibrator = IsotonicRegression(
            y_min=1e-5, y_max=1.0 - 1e-5, increasing=True, out_of_bounds="clip"
        )
        self.global_calibrator.fit(raw_probs, y_true)

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Calibrates input raw probabilities."""
        p_arr = np.clip(raw_probs, 0.0, 1.0)
        if self.global_calibrator is not None:
            return self.global_calibrator.predict(p_arr)

        if self.calibrators:
            # Average predictions across fold calibrators
            cal_preds = np.array([cal.predict(p_arr) for cal in self.calibrators])
            return np.mean(cal_preds, axis=0)

        return p_arr

    def evaluate(self, y_true: np.ndarray, raw_probs: np.ndarray) -> Dict[str, float]:
        """Evaluates calibration quality (ECE, Brier score, Log-loss) before and after."""
        cal_probs = self.calibrate(raw_probs)
        cal_clipped = np.clip(cal_probs, 1e-6, 1.0 - 1e-6)
        raw_clipped = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)

        return {
            "raw_ece": compute_ece(y_true, raw_clipped),
            "calibrated_ece": compute_ece(y_true, cal_clipped),
            "raw_brier": float(brier_score_loss(y_true, raw_clipped)),
            "calibrated_brier": float(brier_score_loss(y_true, cal_clipped)),
            "raw_logloss": float(log_loss(y_true, raw_clipped)),
            "calibrated_logloss": float(log_loss(y_true, cal_clipped)),
        }
