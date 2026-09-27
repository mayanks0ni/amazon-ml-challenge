"""Out-of-fold training pipeline with zero label leakage (Section 14.3 & 21.2)."""
from typing import Dict, List, Optional, Sequence, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from ber.models.lgbm import train_lgbm_fold
from ber.models.stage2b import train_stage2b_fold
from ber.features.group import compute_group_context_features
from ber.training.folds import make_s1_group_folds
from ber.training.hard_negatives import assign_hard_negative_weights
from ber.calibration.isotonic import CrossFittedIsotonicCalibrator
from ber.decision.exclusivity import apply_exclusivity
from ber.decision.expected_f import select_expected_f05
from ber.evaluation.f05 import compute_macro_f05


class CPFCModelPipeline:
    """End-to-end trained CPFC pipeline containing all fold models, calibrator, and decision scalars."""

    def __init__(
        self,
        stage2a_models: List[lgb.Booster],
        stage2b_models: List[lgb.Booster],
        calibrator: CrossFittedIsotonicCalibrator,
        feature_cols_2a: List[str],
        feature_cols_2b: List[str],
        optimal_beta: float = 0.0,
        optimal_lam: float = 0.03,
    ):
        self.stage2a_models = stage2a_models
        self.stage2b_models = stage2b_models
        self.calibrator = calibrator
        self.feature_cols_2a = feature_cols_2a
        self.feature_cols_2b = feature_cols_2b
        self.optimal_beta = optimal_beta
        self.optimal_lam = optimal_lam

    def predict_pairs(
        self,
        df_features_2a: pd.DataFrame,
        pair_keys: List[Tuple[str, str]],
    ) -> Dict[Tuple[str, str], float]:
        """Runs full inference: 2a ensemble -> group features -> 2b ensemble -> isotonic calibration."""
        if len(pair_keys) == 0:
            return {}

        # 1. Stage 2a prediction (average across 5 fold models)
        p2a_preds = np.zeros(len(pair_keys), dtype=np.float64)
        for booster in self.stage2a_models:
            p2a_preds += booster.predict(df_features_2a[self.feature_cols_2a])
        p2a_preds /= len(self.stage2a_models)

        # 2. Stage 2b collective group context features
        group_feats_list = compute_group_context_features(pair_keys, p2a_preds)
        df_group = pd.DataFrame(group_feats_list)

        df_full_2b = pd.concat([df_features_2a.reset_index(drop=True), df_group], axis=1)

        # 3. Stage 2b prediction (average across 5 fold models)
        p2b_preds = np.zeros(len(pair_keys), dtype=np.float64)
        for booster in self.stage2b_models:
            p2b_preds += booster.predict(df_full_2b[self.feature_cols_2b])
        p2b_preds /= len(self.stage2b_models)

        # 4. Cross-fitted isotonic calibration
        cal_probs = self.calibrator.calibrate(p2b_preds)

        return {k: float(p) for k, p in zip(pair_keys, cal_probs)}


def run_oof_training_pipeline(
    df_features_2a: pd.DataFrame,
    pair_keys: List[Tuple[str, str]],
    labels: np.ndarray,
    monotone_constraints: Optional[List[int]] = None,
    ground_truth: Optional[Dict[str, List[str]]] = None,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[CPFCModelPipeline, Dict[str, float]]:
    """Runs 5-fold GroupKFold training for Stage 2a, Stage 2b, and calibration."""
    s1_ids = [k[0] for k in pair_keys]
    folds = make_s1_group_folds(s1_ids, n_splits=n_splits, seed=seed)

    n_pairs = len(pair_keys)
    oof_p2a = np.zeros(n_pairs, dtype=np.float64)
    stage2a_models: List[lgb.Booster] = []

    feature_cols_2a = list(df_features_2a.columns)

    # Compute sample weights for hard negatives
    sample_weights = assign_hard_negative_weights(df_features_2a, labels, multiplier=2.0)

    # --- Step 1: Train Stage 2a Models ---
    for fold_idx, (train_s1, val_s1) in enumerate(folds):
        train_mask = np.array([sid in train_s1 for sid in s1_ids])
        val_mask = np.array([sid in val_s1 for sid in s1_ids])

        x_train = df_features_2a.iloc[train_mask]
        y_train = labels[train_mask]
        w_train = sample_weights[train_mask]

        x_val = df_features_2a.iloc[val_mask]
        y_val = labels[val_mask]

        booster = train_lgbm_fold(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            monotone_constraints=monotone_constraints,
            sample_weight_train=w_train,
            seed=seed + fold_idx,
            n_estimators=3000,
            early_stopping_rounds=150,
        )
        stage2a_models.append(booster)

        # OOF prediction for validation fold
        oof_p2a[val_mask] = booster.predict(x_val)

    # --- Step 2: Compute Group-Context Features on Stage 2a OOF ---
    group_feats_list = compute_group_context_features(pair_keys, oof_p2a)
    df_group = pd.DataFrame(group_feats_list)
    df_full_2b = pd.concat([df_features_2a.reset_index(drop=True), df_group], axis=1)
    feature_cols_2b = list(df_full_2b.columns)

    # --- Step 3: Train Stage 2b Models ---
    oof_p2b = np.zeros(n_pairs, dtype=np.float64)
    stage2b_models: List[lgb.Booster] = []

    for fold_idx, (train_s1, val_s1) in enumerate(folds):
        train_mask = np.array([sid in train_s1 for sid in s1_ids])
        val_mask = np.array([sid in val_s1 for sid in s1_ids])

        x_train = df_full_2b.iloc[train_mask]
        y_train = labels[train_mask]
        x_val = df_full_2b.iloc[val_mask]
        y_val = labels[val_mask]

        booster_2b = train_stage2b_fold(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            seed=seed + fold_idx,
            n_estimators=2000,
            early_stopping_rounds=100,
        )
        stage2b_models.append(booster_2b)
        oof_p2b[val_mask] = booster_2b.predict(x_val)

    # --- Step 4: Fit Isotonic Calibration on OOF ---
    calibrator = CrossFittedIsotonicCalibrator()
    calibrator.fit_global(oof_p2b, labels)
    calibrated_oof_p = calibrator.calibrate(oof_p2b)

    calib_metrics = calibrator.evaluate(labels, oof_p2b)

    # --- Step 5: Tune Decision Shift Beta on OOF ---
    oof_pair_probs = {k: float(p) for k, p in zip(pair_keys, calibrated_oof_p)}

    # Apply soft exclusivity
    excl_probs = apply_exclusivity(oof_pair_probs, delta=0.10, shrink=0.25, hard=False)

    # Group probabilities by S1
    s1_to_pair_candidates: Dict[str, List[Tuple[str, float]]] = {}
    for (sid, pid), p in excl_probs.items():
        if sid not in s1_to_pair_candidates:
            s1_to_pair_candidates[sid] = []
        s1_to_pair_candidates[sid].append((pid, p))

    best_beta = 0.0
    best_lam = 0.01
    best_f05 = -1.0
    best_predictions: Dict[str, List[str]] = {}

    if ground_truth is not None:
        for test_lam in [0.005, 0.01, 0.02, 0.04]:
            for beta in np.linspace(-1.5, 1.5, 31):
                curr_preds: Dict[str, List[str]] = {}
                for sid, cands in s1_to_pair_candidates.items():
                    cands_sorted = sorted(cands, key=lambda x: -x[1])
                    c_ids = [c[0] for c in cands_sorted]
                    c_probs = [c[1] for c in cands_sorted]

                    sel_indices, k, v = select_expected_f05(c_probs, lam=test_lam, beta=float(beta))
                    curr_preds[sid] = [c_ids[i] for i in sel_indices]

                eval_res = compute_macro_f05(curr_preds, ground_truth)
                if eval_res["macro_f05"] > best_f05:
                    best_f05 = eval_res["macro_f05"]
                    best_beta = float(beta)
                    best_lam = test_lam
                    best_predictions = curr_preds
    else:
        for sid, cands in s1_to_pair_candidates.items():
            cands_sorted = sorted(cands, key=lambda x: -x[1])
            c_ids = [c[0] for c in cands_sorted]
            c_probs = [c[1] for c in cands_sorted]
            sel_indices, k, v = select_expected_f05(c_probs, lam=0.01, beta=0.0)
            best_predictions[sid] = [c_ids[i] for i in sel_indices]

    eval_results = {}
    if ground_truth is not None:
        eval_results = compute_macro_f05(best_predictions, ground_truth)
        eval_results.update(calib_metrics)
        eval_results["optimal_beta"] = best_beta
        eval_results["optimal_lam"] = best_lam

    model_pipeline = CPFCModelPipeline(
        stage2a_models=stage2a_models,
        stage2b_models=stage2b_models,
        calibrator=calibrator,
        feature_cols_2a=feature_cols_2a,
        feature_cols_2b=feature_cols_2b,
        optimal_beta=best_beta,
        optimal_lam=best_lam,
    )

    return model_pipeline, eval_results
