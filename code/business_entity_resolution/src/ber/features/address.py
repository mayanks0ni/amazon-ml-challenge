"""Address feature extraction module (Section 13.2)."""
from typing import Dict, List, Optional
import numpy as np
from rapidfuzz import distance, fuzz
from ber.io.schemas import CanonicalRecord
from ber.extraction.numbers import compare_house_numbers, compare_unit_numbers
from ber.extraction.postal import compare_postal_codes
from ber.features.rarity import CorpusIDF, compute_idf_shared_unshared


def extract_address_features(
    rec_a: CanonicalRecord,
    rec_b: CanonicalRecord,
    idf_model: CorpusIDF,
) -> Dict[str, float]:
    """Extracts all ~25 address similarity and discrepancy features (Section 13.2)."""
    feats: Dict[str, float] = {}

    # Exact string equality
    feats["addr_exact_eq"] = 1.0 if rec_a.addr_norm and rec_a.addr_norm == rec_b.addr_norm else 0.0

    # Edit distance metrics (rapidfuzz)
    feats["addr_lev_ratio"] = distance.Levenshtein.normalized_similarity(rec_a.addr_norm, rec_b.addr_norm)
    feats["addr_token_set_ratio"] = fuzz.token_set_ratio(rec_a.addr_norm, rec_b.addr_norm) / 100.0
    feats["addr_token_sort_ratio"] = fuzz.token_sort_ratio(rec_a.addr_norm, rec_b.addr_norm) / 100.0
    feats["addr_fold_sim"] = fuzz.token_set_ratio(rec_a.addr_fold, rec_b.addr_fold) / 100.0

    # Core similarity (excluding landmark)
    feats["addr_core_sim"] = fuzz.token_set_ratio(rec_a.addr_norm, rec_b.addr_norm) / 100.0

    # Postal code comparison
    p_eq, p_both, p_pref = compare_postal_codes(rec_a.postal, rec_b.postal)
    feats["postal_eq"] = float(p_eq)
    feats["postal_both_present"] = float(p_both)
    feats["postal_prefix_len"] = float(p_pref)
    feats["postal_conflict"] = 1.0 if p_both == 1 and p_eq == 0 else 0.0

    # House number / street number (branch discriminator)
    h_eq, h_conflict = compare_house_numbers(rec_a.house_no, rec_b.house_no)
    feats["house_eq"] = float(h_eq)
    feats["house_conflict"] = float(h_conflict)  # Crucial: strong negative monotone constraint

    # Unit / suite / shop number
    u_eq, u_conflict = compare_unit_numbers(rec_a.unit, rec_b.unit)
    feats["unit_eq"] = float(u_eq)
    feats["unit_conflict"] = float(u_conflict)

    # Digit tokens agreement and disagreement (e.g. 12/3 vs Sector-5)
    set_nums_a = set(rec_a.addr_nums)
    set_nums_b = set(rec_b.addr_nums)
    if set_nums_a and set_nums_b:
        inter = len(set_nums_a.intersection(set_nums_b))
        union = len(set_nums_a.union(set_nums_b))
        feats["nums_jaccard"] = float(inter / union) if union > 0 else 0.0
        feats["nums_conflict_count"] = float(len(set_nums_a.symmetric_difference(set_nums_b)))
    else:
        feats["nums_jaccard"] = 0.0
        feats["nums_conflict_count"] = 0.0

    # Tail segment comparison (city/state proxy without external gazetteer)
    feats["tail_sim"] = fuzz.token_set_ratio(rec_a.addr_tail, rec_b.addr_tail) / 100.0
    tail_toks_a = set(rec_a.addr_tail.split())
    tail_toks_b = set(rec_b.addr_tail.split())
    if tail_toks_a and tail_toks_b:
        feats["tail_token_overlap"] = float(len(tail_toks_a.intersection(tail_toks_b)) / len(tail_toks_a.union(tail_toks_b)))
    else:
        feats["tail_token_overlap"] = 0.0

    # Landmark phrase
    feats["landmark_present_a"] = 1.0 if rec_a.landmark else 0.0
    feats["landmark_present_b"] = 1.0 if rec_b.landmark else 0.0
    feats["landmark_sim"] = (
        fuzz.token_set_ratio(rec_a.landmark, rec_b.landmark) / 100.0
        if (rec_a.landmark and rec_b.landmark)
        else 0.0
    )

    # Address rare-token IDF statistics
    addr_idf_stats = compute_idf_shared_unshared(rec_a.addr_tokens, rec_b.addr_tokens, idf_model)
    feats["addr_idf_shared_max"] = addr_idf_stats["idf_shared_max"]
    feats["addr_idf_unshared_sum"] = (
        addr_idf_stats["idf_unshared_sum_a"] + addr_idf_stats["idf_unshared_sum_b"]
    )

    # Length and missingness
    len_a = len(rec_a.addr_norm)
    len_b = len(rec_b.addr_norm)
    max_l = max(len_a, len_b)
    feats["addr_len_ratio"] = (min(len_a, len_b) / max_l) if max_l > 0 else 1.0
    feats["addr_missing_a"] = float(rec_a.addr_missing)
    feats["addr_missing_b"] = float(rec_b.addr_missing)

    return feats
