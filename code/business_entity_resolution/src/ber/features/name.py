"""Name feature extraction module (Section 13.1)."""
from typing import Dict, List, Optional
import numpy as np
from rapidfuzz import distance, fuzz
from ber.io.schemas import CanonicalRecord
from ber.extraction.legal_form import check_legal_form_compatibility
from ber.normalization.mining import compute_abbrev_align
from ber.features.rarity import CorpusIDF, compute_idf_shared_unshared


def compute_soft_tfidf(
    tokens_a: List[str],
    tokens_b: List[str],
    idf_model: CorpusIDF,
    threshold: float = 0.90,
) -> float:
    """Soft TF-IDF (Cohen et al.): sums IDF-weighted products for token pairs with JW >= threshold."""
    if not tokens_a or not tokens_b:
        return 0.0

    score = 0.0
    for ta in tokens_a:
        best_sim = 0.0
        best_tb = ""
        for tb in tokens_b:
            sim = distance.JaroWinkler.similarity(ta, tb)
            if sim > best_sim:
                best_sim = sim
                best_tb = tb
        if best_sim >= threshold:
            score += idf_model.get_idf(ta) * idf_model.get_idf(best_tb) * best_sim

    norm_a = sum(idf_model.get_idf(t) ** 2 for t in tokens_a) ** 0.5
    norm_b = sum(idf_model.get_idf(t) ** 2 for t in tokens_b) ** 0.5
    if norm_a > 0 and norm_b > 0:
        return float(score / (norm_a * norm_b))
    return 0.0


def compute_monge_elkan_jw(tokens_a: List[str], tokens_b: List[str]) -> Dict[str, float]:
    """Computes Monge-Elkan similarity using Jaro-Winkler as inner similarity in both directions."""
    if not tokens_a or not tokens_b:
        return {"monge_elkan_a2b": 0.0, "monge_elkan_b2a": 0.0, "monge_elkan_min": 0.0, "monge_elkan_max": 0.0}

    # Direction A -> B
    best_matches_a = []
    for ta in tokens_a:
        best_sim = max((distance.JaroWinkler.similarity(ta, tb) for tb in tokens_b), default=0.0)
        best_matches_a.append(best_sim)
    score_a2b = float(np.mean(best_matches_a))

    # Direction B -> A
    best_matches_b = []
    for tb in tokens_b:
        best_sim = max((distance.JaroWinkler.similarity(tb, ta) for ta in tokens_a), default=0.0)
        best_matches_b.append(best_sim)
    score_b2a = float(np.mean(best_matches_b))

    return {
        "monge_elkan_a2b": score_a2b,
        "monge_elkan_b2a": score_b2a,
        "monge_elkan_min": min(score_a2b, score_b2a),
        "monge_elkan_max": max(score_a2b, score_b2a),
    }


def extract_name_features(
    rec_a: CanonicalRecord,
    rec_b: CanonicalRecord,
    idf_model: CorpusIDF,
) -> Dict[str, float]:
    """Extracts all ~30 name similarity and discrepancy features (Section 13.1)."""
    feats: Dict[str, float] = {}

    # Exact string equality at norm, core, fold
    feats["name_exact_norm"] = 1.0 if rec_a.name_norm and rec_a.name_norm == rec_b.name_norm else 0.0
    feats["name_exact_core"] = 1.0 if rec_a.name_core and rec_a.name_core == rec_b.name_core else 0.0
    feats["name_exact_fold"] = 1.0 if rec_a.name_fold and rec_a.name_fold == rec_b.name_fold else 0.0

    # Edit distance metrics (rapidfuzz)
    feats["name_lev_ratio_norm"] = distance.Levenshtein.normalized_similarity(rec_a.name_norm, rec_b.name_norm)
    feats["name_lev_ratio_core"] = distance.Levenshtein.normalized_similarity(rec_a.name_core, rec_b.name_core)
    feats["name_damerau_ratio_core"] = distance.DamerauLevenshtein.normalized_similarity(rec_a.name_core, rec_b.name_core)

    # Jaro-Winkler
    feats["name_jw_core"] = distance.JaroWinkler.similarity(rec_a.name_core, rec_b.name_core)
    feats["name_jw_fold"] = distance.JaroWinkler.similarity(rec_a.name_fold, rec_b.name_fold)

    # Token set & sort ratios
    feats["name_token_set_ratio_core"] = fuzz.token_set_ratio(rec_a.name_core, rec_b.name_core) / 100.0
    feats["name_token_sort_ratio_core"] = fuzz.token_sort_ratio(rec_a.name_core, rec_b.name_core) / 100.0
    feats["name_partial_ratio_core"] = fuzz.partial_ratio(rec_a.name_core, rec_b.name_core) / 100.0
    feats["name_token_set_ratio_fold"] = fuzz.token_set_ratio(rec_a.name_fold, rec_b.name_fold) / 100.0

    # Token Jaccard
    set_a = set(rec_a.name_core_tokens)
    set_b = set(rec_b.name_core_tokens)
    union_len = len(set_a.union(set_b))
    feats["name_token_jaccard_core"] = (len(set_a.intersection(set_b)) / union_len) if union_len > 0 else 0.0

    # Soft TF-IDF & Monge-Elkan
    feats["name_soft_tfidf"] = compute_soft_tfidf(rec_a.name_core_tokens, rec_b.name_core_tokens, idf_model)
    me_feats = compute_monge_elkan_jw(rec_a.name_core_tokens, rec_b.name_core_tokens)
    feats.update(me_feats)

    # Rare-token IDF evidence (shared vs unshared)
    idf_stats = compute_idf_shared_unshared(rec_a.name_core_tokens, rec_b.name_core_tokens, idf_model)
    feats.update(idf_stats)

    # First and last token agreement
    feats["name_first_token_eq"] = (
        1.0
        if rec_a.name_core_tokens
        and rec_b.name_core_tokens
        and rec_a.name_core_tokens[0] == rec_b.name_core_tokens[0]
        else 0.0
    )
    feats["name_last_token_eq"] = (
        1.0
        if rec_a.name_core_tokens
        and rec_b.name_core_tokens
        and rec_a.name_core_tokens[-1] == rec_b.name_core_tokens[-1]
        else 0.0
    )

    # Acronym match
    acronym_match = 0.0
    if rec_a.name_acronym and rec_b.name_core_tokens:
        if rec_a.name_acronym in rec_b.name_core_tokens or rec_a.name_acronym == rec_b.name_acronym:
            acronym_match = 1.0
    if rec_b.name_acronym and rec_a.name_core_tokens:
        if rec_b.name_acronym in rec_a.name_core_tokens or rec_b.name_acronym == rec_a.name_acronym:
            acronym_match = 1.0
    feats["name_acronym_match"] = acronym_match

    # Abbreviation alignment (prefix / skeleton match)
    feats["name_abbrev_align"] = compute_abbrev_align(rec_a.name_core_tokens, rec_b.name_core_tokens)

    # Legal form compatibility
    lf_eq, lf_conflict = check_legal_form_compatibility(rec_a.legal_form, rec_b.legal_form)
    feats["legal_form_eq"] = float(lf_eq)
    feats["legal_form_conflict"] = float(lf_conflict)

    # Length ratios and token counts
    len_a = len(rec_a.name_core)
    len_b = len(rec_b.name_core)
    max_len = max(len_a, len_b)
    feats["name_len_ratio_chars"] = (min(len_a, len_b) / max_len) if max_len > 0 else 1.0
    tok_a = len(rec_a.name_core_tokens)
    tok_b = len(rec_b.name_core_tokens)
    max_tok = max(tok_a, tok_b)
    feats["name_len_ratio_tokens"] = (min(tok_a, tok_b) / max_tok) if max_tok > 0 else 1.0
    feats["name_n_tokens_a"] = float(tok_a)
    feats["name_n_tokens_b"] = float(tok_b)

    # Digits inside names (7-Eleven, 24/7)
    digits_a = set(c for c in rec_a.name_raw if c.isdigit())
    digits_b = set(c for c in rec_b.name_raw if c.isdigit())
    if digits_a and digits_b:
        feats["digits_in_name_eq"] = 1.0 if digits_a == digits_b else 0.0
        feats["digits_in_name_conflict"] = 1.0 if digits_a != digits_b else 0.0
    else:
        feats["digits_in_name_eq"] = 0.0
        feats["digits_in_name_conflict"] = 0.0

    # Alternate name / DBA maximum similarity
    alt_sims = [0.0]
    for alt_a in [rec_a.name_core] + rec_a.name_alts:
        for alt_b in [rec_b.name_core] + rec_b.name_alts:
            alt_sims.append(fuzz.token_set_ratio(alt_a, alt_b) / 100.0)
    feats["alt_name_max_sim"] = max(alt_sims)

    return feats
