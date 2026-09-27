"""Cross-field, retrieval provenance, and ambiguity features (Section 13.3)."""
from typing import Dict, Optional
from ber.io.schemas import CanonicalRecord, CandidateProvenance


def extract_cross_features(
    rec_a: CanonicalRecord,
    rec_b: CanonicalRecord,
    prov: Optional[CandidateProvenance],
    cand_count_i: int,
    name_sim: float,
    addr_sim: float,
    freq_stats: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Extracts cross-field interactions, provenance bits, and ambiguity metrics."""
    feats: Dict[str, float] = {}

    # Country equality (only country feature - Rule O8 open-set compliant)
    feats["same_country"] = (
        1.0
        if rec_a.country_norm and rec_b.country_norm and rec_a.country_norm == rec_b.country_norm
        else 0.0
    )

    # Name tokens inside the other address (e.g. building/mall names: "Nike at Inorbit Mall")
    set_name_a = set(rec_a.name_core_tokens)
    set_name_b = set(rec_b.name_core_tokens)
    set_addr_a = set(rec_a.addr_tokens)
    set_addr_b = set(rec_b.addr_tokens)

    feats["name_in_addr_a2b"] = (
        float(len(set_name_a.intersection(set_addr_b)) / len(set_name_a)) if set_name_a else 0.0
    )
    feats["name_in_addr_b2a"] = (
        float(len(set_name_b.intersection(set_addr_a)) / len(set_name_b)) if set_name_b else 0.0
    )

    # Source indicator
    feats["is_s3"] = 1.0 if rec_b.source == "S3" else 0.0

    # Provenance channel features
    if prov is not None:
        feats["channel_mask"] = float(prov.mask)
        feats["n_channels"] = float(bin(prov.mask).count("1"))
        feats["rank_c1"] = float(prov.rank.get("c1_exact", 999))
        feats["rank_c2"] = float(prov.rank.get("c2_tfidf", 999))
        feats["rank_c3"] = float(prov.rank.get("c3_rare", 999))
        feats["rank_c4"] = float(prov.rank.get("c4_phonetic", 999))
        feats["rank_c5"] = float(prov.rank.get("c5_address", 999))
        feats["rank_c6"] = float(prov.rank.get("c6_snm", 999))
    else:
        feats["channel_mask"] = 0.0
        feats["n_channels"] = 0.0
        feats["rank_c1"] = 999.0
        feats["rank_c2"] = 999.0
        feats["rank_c3"] = 999.0
        feats["rank_c4"] = 999.0
        feats["rank_c5"] = 999.0
        feats["rank_c6"] = 999.0

    # Candidate set density (normalized)
    feats["cand_count_i"] = float(cand_count_i)

    # Frequencies from corpus
    if freq_stats:
        feats["name_core_freq_s1"] = freq_stats.get(f"name_s1_{rec_a.name_core}", 1.0)
        feats["name_core_freq_pool"] = freq_stats.get(f"name_pool_{rec_b.name_core}", 1.0)
        feats["addr_freq_pool"] = freq_stats.get(f"addr_pool_{rec_b.addr_norm}", 1.0)
    else:
        feats["name_core_freq_s1"] = 1.0
        feats["name_core_freq_pool"] = 1.0
        feats["addr_freq_pool"] = 1.0

    # Non-linear interaction features flagging branches vs co-locations
    # Branches: high name similarity, low address similarity
    # Co-locations: low name similarity, high address similarity
    feats["name_sim_x_addr_sim"] = name_sim * addr_sim
    feats["name_hi_addr_lo"] = 1.0 if (name_sim >= 0.85 and addr_sim < 0.40) else 0.0
    feats["name_lo_addr_hi"] = 1.0 if (name_sim < 0.40 and addr_sim >= 0.85) else 0.0

    return feats
