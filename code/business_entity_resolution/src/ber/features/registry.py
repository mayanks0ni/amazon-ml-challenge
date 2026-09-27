"""Feature registry, column ordering, and monotone constraints (Section 13 & 14.2)."""
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from ber.io.schemas import CanonicalRecord, CandidateProvenance
from ber.features.name import extract_name_features
from ber.features.address import extract_address_features
from ber.features.cross import extract_cross_features
from ber.features.rarity import CorpusIDF

# Monotone constraints definition from Section 14.2:
# +1: Positive indicator of match (e.g. name similarity, postal equality)
# -1: Negative indicator / conflict (e.g. house conflict, unshared rare tokens)
#  0: Neutral / unconstrained
MONOTONE_SPECS = {
    # Name similarities
    "name_exact_norm": 1,
    "name_exact_core": 1,
    "name_exact_fold": 1,
    "name_lev_ratio_core": 1,
    "name_jw_core": 1,
    "name_token_set_ratio_core": 1,
    "name_soft_tfidf": 1,
    "idf_shared_max": 1,
    "idf_shared_sum": 1,
    "acronym_match": 1,
    "legal_form_eq": 1,
    "alt_name_max_sim": 1,
    # Name conflicts
    "legal_form_conflict": -1,
    "idf_unshared_max": -1,
    "digits_in_name_conflict": -1,
    # Address similarities
    "addr_exact_eq": 1,
    "addr_token_set_ratio": 1,
    "postal_eq": 1,
    "house_eq": 1,
    "unit_eq": 1,
    "nums_jaccard": 1,
    "addr_idf_shared_max": 1,
    # Address conflicts
    "house_conflict": -1,
    "unit_conflict": -1,
    "postal_conflict": -1,
    "addr_idf_unshared_sum": -1,
    # Cross
    "same_country": 1,
    "name_sim_x_addr_sim": 1,
    "name_hi_addr_lo": -1,  # branch indicator
}


def build_pair_feature_vector(
    rec_a: CanonicalRecord,
    rec_b: CanonicalRecord,
    prov: Optional[CandidateProvenance],
    cand_count_i: int,
    idf_model: CorpusIDF,
) -> Dict[str, float]:
    """Extracts complete feature dictionary for a single candidate pair."""
    name_feats = extract_name_features(rec_a, rec_b, idf_model)
    addr_feats = extract_address_features(rec_a, rec_b, idf_model)

    cross_feats = extract_cross_features(
        rec_a=rec_a,
        rec_b=rec_b,
        prov=prov,
        cand_count_i=cand_count_i,
        name_sim=name_feats.get("name_token_set_ratio_core", 0.0),
        addr_sim=addr_feats.get("addr_token_set_ratio", 0.0),
    )

    all_feats = {}
    all_feats.update(name_feats)
    all_feats.update(addr_feats)
    all_feats.update(cross_feats)
    return all_feats


def extract_features_for_candidates(
    candidate_dict: Dict[str, List[str]],
    s1_dict: Dict[str, CanonicalRecord],
    pool_dict: Dict[str, CanonicalRecord],
    provenance_dict: Optional[Dict[str, Dict[str, CandidateProvenance]]] = None,
    idf_model: Optional[CorpusIDF] = None,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]], List[str]]:
    """Builds tabular feature matrix for all pairs in candidate_dict.

    Returns:
        (df_features, pair_keys, feature_columns)
    """
    if idf_model is None:
        idf_model = CorpusIDF(list(s1_dict.values()) + list(pool_dict.values()))

    pair_keys: List[Tuple[str, str]] = []
    rows: List[Dict[str, float]] = []

    for s1_id, pool_ids in candidate_dict.items():
        s1_rec = s1_dict.get(s1_id)
        if not s1_rec:
            continue

        cand_count = len(pool_ids)
        prov_map = provenance_dict.get(s1_id, {}) if provenance_dict else {}

        for pool_id in pool_ids:
            pool_rec = pool_dict.get(pool_id)
            if not pool_rec:
                continue

            prov = prov_map.get(pool_id)
            feat_vec = build_pair_feature_vector(
                rec_a=s1_rec,
                rec_b=pool_rec,
                prov=prov,
                cand_count_i=cand_count,
                idf_model=idf_model,
            )

            pair_keys.append((s1_id, pool_id))
            rows.append(feat_vec)

    if not rows:
        return pd.DataFrame(), [], []

    df = pd.DataFrame(rows)
    feature_cols = sorted(df.columns.tolist())
    df = df[feature_cols]

    return df, pair_keys, feature_cols


def get_monotone_constraints(feature_cols: Sequence[str]) -> List[int]:
    """Returns monotone constraint list aligned with the feature columns."""
    return [MONOTONE_SPECS.get(col, 0) for col in feature_cols]
