"""Leave-One-Country-Out (LOCO) validation proxy for France (Section 21.1)."""
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from ber.evaluation.f05 import compute_macro_f05


def split_loco_indices(
    pair_keys: Sequence[Tuple[str, str]],
    s1_countries: Dict[str, str],
) -> Dict[str, Tuple[List[int], List[int]]]:
    """Partitions dataset into LOCO train and test splits (e.g. US <-> India)."""
    splits: Dict[str, Tuple[List[int], List[int]]] = {}

    countries = set(s1_countries.values())
    for target_country in countries:
        if not target_country:
            continue

        train_indices: List[int] = []
        eval_indices: List[int] = []

        for idx, (sid, _) in enumerate(pair_keys):
            c = s1_countries.get(sid, "")
            if c == target_country:
                eval_indices.append(idx)
            else:
                train_indices.append(idx)

        if train_indices and eval_indices:
            splits[target_country] = (train_indices, eval_indices)

    return splits
