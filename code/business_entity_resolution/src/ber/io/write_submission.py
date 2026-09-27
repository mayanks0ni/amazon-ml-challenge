"""Submission file writer strictly compliant with Section 27.1 specifications."""
import csv
import os
from typing import Collection, Dict, List, Optional, Sequence, Set, Tuple


def write_lists(
    path: str,
    header2: str,
    s1_ids: Sequence[str],
    lists: Dict[str, Sequence[str]],
    valid_pool_ids: Optional[Set[str]] = None,
) -> None:
    """Writes TSV submission list according to competition rules.

    Args:
        path: Path to write TSV file to.
        header2: Second column header name ('candidate_entity_ids' or 'matched_entity_ids').
        s1_ids: All test S1 IDs in their exact original file order.
        lists: Dict mapping s1_id -> sequence of predicted or candidate IDs.
        valid_pool_ids: Set of valid S2/S3 IDs from the test set for integrity verification.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(
            f,
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
            escapechar=None,
            lineterminator="\n",
        )
        # Header row
        w.writerow(["source1_entity_id", header2])

        seen_s1: Set[str] = set()
        for sid in s1_ids:
            if sid in seen_s1:
                raise ValueError(f"Duplicate source1_entity_id encountered: {sid}")
            seen_s1.add(sid)

            seen_target: Set[str] = set()
            out_ids: List[str] = []

            for rid in lists.get(sid, []):
                rid_clean = str(rid).strip()
                if not rid_clean:
                    continue

                # Assert prefix is S2- or S3-
                if not (rid_clean.startswith("S2-") or rid_clean.startswith("S3-")):
                    raise ValueError(f"Invalid pool record ID prefix (must be S2- or S3-): {rid_clean}")

                # Assert membership in valid test pool if provided
                if valid_pool_ids is not None and rid_clean not in valid_pool_ids:
                    raise ValueError(f"Record {rid_clean} is not present in the valid test pool!")

                # Order-preserving deduplication per list
                if rid_clean not in seen_target:
                    seen_target.add(rid_clean)
                    out_ids.append(rid_clean)

            # Write joined list (empty string for singletons)
            w.writerow([sid, ",".join(out_ids)])


def write_submission_outputs(
    test_s1_ids: Sequence[str],
    candidates: Dict[str, Sequence[str]],
    matches: Dict[str, Sequence[str]],
    valid_pool_ids: Set[str],
    output_dir: str = "output",
) -> Tuple[str, str]:
    """Writes both matching_results.tsv and candidate_pairs.tsv and enforces contracts.

    Checks that matches are a subset of candidates for every S1.
    """
    for sid in test_s1_ids:
        cand_set = set(candidates.get(sid, []))
        match_list = matches.get(sid, [])
        for m in match_list:
            if m not in cand_set:
                raise ValueError(
                    f"Contract violation for {sid}: matched ID {m} is not in candidates!"
                )

    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")

    write_lists(
        cand_path,
        "candidate_entity_ids",
        test_s1_ids,
        candidates,
        valid_pool_ids,
    )

    write_lists(
        match_path,
        "matched_entity_ids",
        test_s1_ids,
        matches,
        valid_pool_ids,
    )

    return match_path, cand_path
