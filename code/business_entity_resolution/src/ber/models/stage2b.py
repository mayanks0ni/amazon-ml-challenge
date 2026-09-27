"""Collective Stage 2b LightGBM with group-context features (Section 14.3)."""
from typing import Dict, List, Optional, Sequence
import lightgbm as lgb
import numpy as np
import pandas as pd


def train_stage2b_fold(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    seed: int = 42,
    n_estimators: int = 1500,
    early_stopping_rounds: int = 100,
) -> lgb.Booster:
    """Trains a single fold collective Stage 2b booster."""
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": max(5, int(len(x_train) * 0.01)) if len(x_train) < 5000 else 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": seed,
        "deterministic": True,
        "verbosity": -1,
        "n_jobs": -1,
    }

    dtrain = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
    dval = lgb.Dataset(x_val, label=y_val, reference=dtrain, free_raw_data=False)

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
