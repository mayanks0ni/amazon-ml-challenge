"""GroupKFold cross-validation splitting by Source 1 entity (Section 21.1)."""
from typing import Dict, Generator, List, Sequence, Tuple
import numpy as np
from sklearn.model_selection import GroupKFold


def make_s1_group_folds(
    s1_ids: Sequence[str],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Creates GroupKFold splits strictly over unique S1 entity IDs.
    Guarantees no S1 entity crosses between train and validation folds.
    """
    unique_s1 = np.array(sorted(list(set(s1_ids))))
    gkf = GroupKFold(n_splits=n_splits)

    dummy_x = np.zeros(len(unique_s1))
    folds = []
    for train_idx, val_idx in gkf.split(dummy_x, groups=unique_s1):
        train_s1 = set(unique_s1[train_idx])
        val_s1 = set(unique_s1[val_idx])
        folds.append((train_s1, val_s1))

    return folds
