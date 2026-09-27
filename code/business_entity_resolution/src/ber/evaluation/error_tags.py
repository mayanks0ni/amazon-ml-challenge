"""Automatic error categorization and tagging (Section 23)."""
from collections import defaultdict
from typing import Dict, List, Sequence, Set
from ber.io.schemas import CanonicalRecord


def tag_errors(
    predictions: Dict[str, Sequence[str]],
    candidates: Dict[str, Sequence[str]],
    ground_truth: Dict[str, Sequence[str]],
    s1_records: Dict[str, CanonicalRecord],
    pool_records: Dict[str, CanonicalRecord],
) -> Dict[str, List[str]]:
    """Automatically categorizes all validation errors into Section 23 error taxonomy."""
    error_buckets: Dict[str, List[str]] = defaultdict(list)

    for s1_id, gold_list in ground_truth.items():
        gold_set = set(gold_list)
        pred_set = set(predictions.get(s1_id, []))
        cand_set = set(candidates.get(s1_id, []))

        s1_rec = s1_records.get(s1_id)

        # Category 11: Singleton false positive
        if len(gold_set) == 0 and len(pred_set) > 0:
            error_buckets["singleton_false_positive"].append(s1_id)

        # Category 1: False merge
        false_merges = pred_set - gold_set
        for fm in false_merges:
            error_buckets["false_merge"].append(f"{s1_id}:{fm}")

            p_rec = pool_records.get(fm)
            if s1_rec and p_rec:
                # Category 4: Same name, other branch
                if s1_rec.name_core == p_rec.name_core and s1_rec.house_no != p_rec.house_no:
                    error_buckets["same_name_other_branch"].append(f"{s1_id}:{fm}")

                # Category 5: Same address, other business
                if s1_rec.addr_norm == p_rec.addr_norm and s1_rec.name_core != p_rec.name_core:
                    error_buckets["same_addr_other_business"].append(f"{s1_id}:{fm}")

        # Category 2 & 3: Missed matches
        missed_matches = gold_set - pred_set
        for mm in missed_matches:
            if mm in cand_set:
                error_buckets["missed_match_in_candidates"].append(f"{s1_id}:{mm}")
            else:
                error_buckets["blocking_failure"].append(f"{s1_id}:{mm}")

    return dict(error_buckets)
