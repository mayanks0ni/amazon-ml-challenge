"""Hard negative mining and sample weight computation (Section 15)."""
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from ber.io.schemas import CanonicalRecord


def assign_hard_negative_weights(
    df_features: pd.DataFrame,
    labels: np.ndarray,
    multiplier: float = 2.0,
) -> np.ndarray:
    """Computes sample weights for training pairs based on Section 15 taxonomy:
    - Positive pairs: weight = 1.0
    - HN1 (Blocked negatives): weight = 1.0
    - HN2 (Same name, different address): weight = 2.0 (name_hi_addr_lo == 1.0)
    - HN3 (Same address, different name): weight = 2.0 (name_lo_addr_hi == 1.0)
    - HN4 / HN5 / HN6 (Name collisions): weight = 1.5
    """
    n_samples = len(labels)
    weights = np.ones(n_samples, dtype=np.float32)

    for i in range(n_samples):
        if labels[i] == 1:
            weights[i] = 1.0
            continue

        row = df_features.iloc[i]
        w = 1.0

        # HN2: Branch collisions (high name sim, low address sim)
        if row.get("name_hi_addr_lo", 0.0) == 1.0:
            w = max(w, multiplier)

        # HN3: Co-location collisions (low name sim, high address sim)
        if row.get("name_lo_addr_hi", 0.0) == 1.0:
            w = max(w, multiplier)

        # House conflict on negatives
        if row.get("house_conflict", 0.0) == 1.0:
            w = max(w, multiplier)

        # Name collisions (exact fold or acronym equal)
        if row.get("name_exact_fold", 0.0) == 1.0 or row.get("name_acronym_match", 0.0) == 1.0:
            w = max(w, 1.5)

        weights[i] = w

    return weights
