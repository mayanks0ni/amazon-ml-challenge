"""Primary LightGBM pair matcher configuration (Section 14.2)."""
from typing import Dict, List, Optional, Sequence
import lightgbm as lgb
import numpy as np
import pandas as pd


def get_default_lgbm_params(
    monotone_constraints: Optional[Sequence[int]] = None,
    seed: int = 42,
) -> Dict[str, any]:
    """Returns official competition LightGBM parameters from Section 14.2."""
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "seed": seed,
        "deterministic": True,
        "verbosity": -1,
        "n_jobs": -1,
    }

    if monotone_constraints is not None:
        params["monotone_constraints"] = list(monotone_constraints)
        params["monotone_constraints_method"] = "advanced"

    return params


def train_lgbm_fold(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    monotone_constraints: Optional[Sequence[int]] = None,
    sample_weight_train: Optional[np.ndarray] = None,
    seed: int = 42,
    n_estimators: int = 2000,
    early_stopping_rounds: int = 150,
) -> lgb.Booster:
    """Trains a single fold LightGBM booster with early stopping and monotone constraints."""
    params = get_default_lgbm_params(monotone_constraints=monotone_constraints, seed=seed)
    # Adapt min_data_in_leaf if training size is smaller than default
    if len(x_train) < 5000:
        params["min_data_in_leaf"] = max(5, int(len(x_train) * 0.01))

    dtrain = lgb.Dataset(
        x_train,
        label=y_train,
        weight=sample_weight_train,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        x_val,
        label=y_val,
        reference=dtrain,
        free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
    ]

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    return booster
