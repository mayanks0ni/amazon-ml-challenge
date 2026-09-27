"""Multi-channel candidate retrieval and blocking pipeline."""
from typing import Dict, List, Optional, Sequence, Tuple
from ber.io.schemas import CanonicalRecord, CandidateProvenance
from ber.blocking.exact import retrieve_c1_exact
from ber.blocking.tfidf_topk import retrieve_c2_tfidf
from ber.blocking.rare_token import retrieve_c3_rare_tokens
from ber.blocking.phonetic import retrieve_c4_phonetic_acronym
from ber.blocking.address import retrieve_c5_address_numeric
from ber.blocking.snm import retrieve_c6_snm
from ber.blocking.embedding import retrieve_c7_embedding
from ber.blocking.token_overlap import retrieve_c8_token_overlap
from ber.blocking.substring import retrieve_c9_substring
from ber.blocking.union import build_candidate_union, s2_s3_candidate_expansion
from ber.blocking.prune import prune_candidates
from ber.blocking.metrics import compute_blocking_metrics


def run_blocking_pipeline(
    s1_records: Sequence[CanonicalRecord],
    pool_records: Sequence[CanonicalRecord],
    budget_m: int = 100,
    enable_c7_embedding: bool = False,
    enable_s2_s3_expansion: bool = True,
    partition_by_country: bool = True,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, CandidateProvenance]]]:
    """Runs full multi-channel candidate blocking cascade.

    Supports country-partitioned execution (Gate G2): runs channels inside country
    partitions to maximize recall and efficiency.  S1 records with unknown country
    are run against the full pool as a fallback.

    Returns:
        (pruned_candidates, candidate_provenance)
        where pruned_candidates is dict mapping s1_id -> list of candidate pool IDs.
    """
    if not s1_records or not pool_records:
        return {r.entity_id: [] for r in s1_records}, {}

    if partition_by_country:
        # Group records by normalized country
        country_groups: Dict[str, Tuple[List[CanonicalRecord], List[CanonicalRecord]]] = {}
        unknown_s1: List[CanonicalRecord] = []

        for r in s1_records:
            c = r.country_norm or ""
            if not c or c == "unknown":
                unknown_s1.append(r)
            else:
                if c not in country_groups:
                    country_groups[c] = ([], [])
                country_groups[c][0].append(r)

        for r in pool_records:
            c = r.country_norm or ""
            if c and c != "unknown" and c in country_groups:
                country_groups[c][1].append(r)
            else:
                # Pool records with unknown/empty country go into ALL groups
                # so they can be found by any S1 regardless of country
                for group in country_groups.values():
                    group[1].append(r)

        all_pruned: Dict[str, List[str]] = {}
        all_prov: Dict[str, Dict[str, CandidateProvenance]] = {}

        for c, (s1_group, pool_group) in country_groups.items():
            if not s1_group:
                continue
            if not pool_group:
                # Fallback to full pool if country partition has no pool records
                pool_group = list(pool_records)

            c_pruned, c_prov = _run_blocking_unpartitioned(
                s1_group,
                pool_group,
                budget_m=budget_m,
                enable_c7_embedding=enable_c7_embedding,
                enable_s2_s3_expansion=enable_s2_s3_expansion,
            )
            all_pruned.update(c_pruned)
            all_prov.update(c_prov)

        # Fallback: run unknown-country S1 records against the full pool
        if unknown_s1:
            u_pruned, u_prov = _run_blocking_unpartitioned(
                unknown_s1,
                list(pool_records),
                budget_m=budget_m,
                enable_c7_embedding=enable_c7_embedding,
                enable_s2_s3_expansion=enable_s2_s3_expansion,
            )
            all_pruned.update(u_pruned)
            all_prov.update(u_prov)

        return all_pruned, all_prov
    else:
        return _run_blocking_unpartitioned(
            s1_records,
            pool_records,
            budget_m=budget_m,
            enable_c7_embedding=enable_c7_embedding,
            enable_s2_s3_expansion=enable_s2_s3_expansion,
        )


def _run_blocking_unpartitioned(
    s1_records: Sequence[CanonicalRecord],
    pool_records: Sequence[CanonicalRecord],
    budget_m: int = 100,
    enable_c7_embedding: bool = False,
    enable_s2_s3_expansion: bool = True,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, CandidateProvenance]]]:
    """Runs all 9 retrieval channels on a record partition."""
    channel_outputs = []

    # Channel C1: Exact keys (bit 0)
    c1 = retrieve_c1_exact(s1_records, pool_records)
    channel_outputs.append(("c1_exact", 0, c1))

    # Channel C2: Name char-TFIDF (bit 1)
    c2 = retrieve_c2_tfidf(s1_records, pool_records, top_k_per_source=50)
    channel_outputs.append(("c2_tfidf", 1, c2))

    # Channel C3: Rare tokens (bit 2)
    c3 = retrieve_c3_rare_tokens(s1_records, pool_records, top_k=50, max_df=200)
    channel_outputs.append(("c3_rare", 2, c3))

    # Channel C4: Phonetic / acronym (bit 3)
    c4 = retrieve_c4_phonetic_acronym(s1_records, pool_records, top_k=15)
    channel_outputs.append(("c4_phonetic", 3, c4))

    # Channel C5: Address numeric / exact / token IDF (bit 4)
    c5 = retrieve_c5_address_numeric(s1_records, pool_records, top_k=30)
    channel_outputs.append(("c5_address", 4, c5))

    # Channel C6: Sorted neighbourhood (bit 5)
    c6 = retrieve_c6_snm(s1_records, pool_records, window=20)
    channel_outputs.append(("c6_snm", 5, c6))

    # Channel C7: Gated embedding (bit 6)
    if enable_c7_embedding:
        c7 = retrieve_c7_embedding(s1_records, pool_records, enabled=True, top_k=20)
        channel_outputs.append(("c7_emb", 6, c7))

    # Channel C8: Token-overlap (bit 7)
    c8 = retrieve_c8_token_overlap(s1_records, pool_records, top_k=30, min_shared=1)
    channel_outputs.append(("c8_token_overlap", 8, c8))

    # Channel C9: Substring / containment (bit 9)
    c9 = retrieve_c9_substring(s1_records, pool_records, top_k=20)
    channel_outputs.append(("c9_substring", 9, c9))

    # Union + Provenance + S2-S3 Expansion
    union_cands = build_candidate_union(
        channel_outputs, pool_records, enable_expansion=enable_s2_s3_expansion
    )

    # Budgeted Pruning
    pruned = prune_candidates(union_cands, budget_m=budget_m, keep_if_channels_ge=2)

    # Ensure every S1 record has an entry in pruned
    for s1 in s1_records:
        if s1.entity_id not in pruned:
            pruned[s1.entity_id] = []

    return pruned, union_cands


__all__ = [
    "run_blocking_pipeline",
    "retrieve_c1_exact",
    "retrieve_c2_tfidf",
    "retrieve_c3_rare_tokens",
    "retrieve_c4_phonetic_acronym",
    "retrieve_c5_address_numeric",
    "retrieve_c6_snm",
    "retrieve_c7_embedding",
    "retrieve_c8_token_overlap",
    "retrieve_c9_substring",
    "build_candidate_union",
    "prune_candidates",
    "compute_blocking_metrics",
]

