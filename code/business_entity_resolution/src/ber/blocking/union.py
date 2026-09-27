"""Union of candidate channels with provenance tracking and S2-S3 expansion (Section 12.1 & 12.4)."""
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple
from rapidfuzz import fuzz
from ber.io.schemas import CanonicalRecord, CandidateProvenance


def s2_s3_candidate_expansion(
    candidates: Dict[str, Dict[str, CandidateProvenance]],
    pool_records: Sequence[CanonicalRecord],
    k: int = 3,
    min_sim: float = 0.90,
) -> Dict[str, Dict[str, CandidateProvenance]]:
    """S2 <-> S3 candidate expansion (Section 12.4).

    If record r in S2 is a strong candidate for S1 (top-3) and record r' in S3
    is a near-duplicate of r (fuzz ratio >= 90 on name), then r' is added to S1's candidates.
    Recovers records that match their sibling better than S1 itself.
    """
    id_to_record = {r.entity_id: r for r in pool_records}

    # Pre-index pool records by source and core name for fast O(1) sibling lookup
    s2_core_index = defaultdict(list)
    s3_core_index = defaultdict(list)
    for r in pool_records:
        if r.name_core:
            if r.source == "S2":
                s2_core_index[r.name_core].append(r)
            elif r.source == "S3":
                s3_core_index[r.name_core].append(r)

    for s1_id, pool_dict in candidates.items():
        if not pool_dict:
            continue
        # Find top-k candidates for this S1
        sorted_cands = sorted(
            pool_dict.items(),
            key=lambda item: min(item[1].rank.values()) if item[1].rank else 999,
        )[:k]

        for cand_id, prov in sorted_cands:
            cand_rec = id_to_record.get(cand_id)
            if not cand_rec or not cand_rec.name_core:
                continue

            target_index = s3_core_index if cand_rec.source == "S2" else s2_core_index
            potential_siblings = target_index.get(cand_rec.name_core, [])

            for other_rec in potential_siblings:
                if other_rec.entity_id in pool_dict:
                    continue

                # Fast similarity on normalized name
                sim = fuzz.token_set_ratio(cand_rec.name_norm, other_rec.name_norm) / 100.0
                if sim >= min_sim:
                    # Add as expanded candidate
                    new_prov = CandidateProvenance(
                        mask=1 << 7,  # Bit 7 indicates S2-S3 sibling expansion
                        rank={"expansion": 1},
                        score={"expansion": sim},
                    )
                    pool_dict[other_rec.entity_id] = new_prov

    return candidates


def build_candidate_union(
    channel_outputs: List[Tuple[str, int, List[Tuple[str, str, int, float]]]],
    pool_records: Sequence[CanonicalRecord],
    enable_expansion: bool = True,
) -> Dict[str, Dict[str, CandidateProvenance]]:
    """Builds unified candidate mapping with provenance tracking across all channels.

    Args:
        channel_outputs: List of (channel_name, channel_bit, [(s1_id, pool_id, rank, score)])
        pool_records: Complete pool of S2 and S3 records
        enable_expansion: Whether to perform S2 <-> S3 expansion

    Returns:
        Dict mapping s1_id -> {pool_id: CandidateProvenance}
    """
    candidates: Dict[str, Dict[str, CandidateProvenance]] = defaultdict(dict)

    for ch_name, ch_bit, records in channel_outputs:
        for s1_id, pool_id, rank, score in records:
            prov = candidates[s1_id].setdefault(pool_id, CandidateProvenance())
            prov.mask |= (1 << ch_bit)
            prov.rank[ch_name] = min(prov.rank.get(ch_name, 999999), rank)
            prov.score[ch_name] = max(prov.score.get(ch_name, 0.0), score)

    if enable_expansion:
        candidates = s2_s3_candidate_expansion(candidates, pool_records)

    return candidates
