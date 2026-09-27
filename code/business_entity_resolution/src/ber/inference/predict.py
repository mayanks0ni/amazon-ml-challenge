"""End-to-end inference pipeline on test split."""
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple
import pandas as pd
from ber.io.read_tsv import load_source_records
from ber.io.write_submission import write_submission_outputs
from ber.normalization.pipeline import canonicalize_record
from ber.blocking import run_blocking_pipeline
from ber.features.registry import extract_features_for_candidates
from ber.features.rarity import CorpusIDF
from ber.training.oof import CPFCModelPipeline
from ber.decision.exclusivity import apply_exclusivity
from ber.decision.expected_f import select_expected_f05


def run_test_inference(
    test_dir: str,
    model_pipeline: CPFCModelPipeline,
    output_dir: str = "output",
    budget_m: int = 100,
    enable_c7: bool = False,
) -> Tuple[str, str]:
    """Executes full end-to-end inference on test data and produces submission TSVs."""
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    # Step 1: Ingest test records
    raw_s1 = load_source_records(s1_path, source_name="S1", split="test")
    raw_s2 = load_source_records(s2_path, source_name="S2", split="test")
    raw_s3 = load_source_records(s3_path, source_name="S3", split="test")

    raw_pool = raw_s2 + raw_s3
    test_s1_ids = [r.entity_id for r in raw_s1]
    valid_pool_ids = set(r.entity_id for r in raw_pool)

    # Step 2: Canonical normalization
    can_s1 = [canonicalize_record(r) for r in raw_s1]
    can_pool = [canonicalize_record(r) for r in raw_pool]

    s1_dict = {r.entity_id: r for r in can_s1}
    pool_dict = {r.entity_id: r for r in can_pool}

    # Step 3: Multi-channel blocking
    pruned_cands, cand_provenance = run_blocking_pipeline(
        s1_records=can_s1,
        pool_records=can_pool,
        budget_m=budget_m,
        enable_c7_embedding=enable_c7,
        partition_by_country=True,
    )

    # Step 4: Extract features for candidate pairs
    idf_model = CorpusIDF(can_s1 + can_pool)
    df_features_2a, pair_keys, _ = extract_features_for_candidates(
        candidate_dict=pruned_cands,
        s1_dict=s1_dict,
        pool_dict=pool_dict,
        provenance_dict=cand_provenance,
        idf_model=idf_model,
    )

    # Step 5: Score pairs with trained CPFC cascade
    calibrated_pair_probs = model_pipeline.predict_pairs(df_features_2a, pair_keys)

    # Step 6: Apply exclusivity
    excl_probs = apply_exclusivity(calibrated_pair_probs, delta=0.10, shrink=0.25, hard=False)

    # Step 7: Expected-F0.5 set selection
    s1_to_pair_candidates: Dict[str, List[Tuple[str, float]]] = {sid: [] for sid in test_s1_ids}
    for (sid, pid), p in excl_probs.items():
        s1_to_pair_candidates[sid].append((pid, p))

    final_matches: Dict[str, List[str]] = {}
    for sid in test_s1_ids:
        cands = s1_to_pair_candidates.get(sid, [])
        if not cands:
            final_matches[sid] = []
            continue

        cands_sorted = sorted(cands, key=lambda x: -x[1])
        c_ids = [c[0] for c in cands_sorted]
        c_probs = [c[1] for c in cands_sorted]

        sel_indices, k, v = select_expected_f05(
            c_probs, lam=model_pipeline.optimal_lam, beta=model_pipeline.optimal_beta
        )
        final_matches[sid] = [c_ids[i] for i in sel_indices]

    # Step 8: Write submission files and assert contracts
    match_path, cand_path = write_submission_outputs(
        test_s1_ids=test_s1_ids,
        candidates=pruned_cands,
        matches=final_matches,
        valid_pool_ids=valid_pool_ids,
        output_dir=output_dir,
    )

    return match_path, cand_path
