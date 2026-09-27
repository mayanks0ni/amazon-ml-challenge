"""Evaluates the trained LightGBM CPFC pipeline (trained_cpfc_pipeline.pkl) on real ground truth."""
import csv
import os
import pickle
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from ber.io.schemas import RawRecord
from ber.normalization.pipeline import canonicalize_record
from ber.blocking import run_blocking_pipeline, compute_blocking_metrics
from ber.features.registry import extract_features_for_candidates
from ber.features.rarity import CorpusIDF
from ber.decision.exclusivity import apply_exclusivity
from ber.decision.expected_f import select_expected_f05
from ber.decision.singleton import evaluate_singletons
from ber.evaluation.f05 import compute_macro_f05
from ber.evaluation.error_tags import tag_errors


def evaluate_trained_model(n_val: int = 2500, data_dir: str = "student_resource/dataset/train"):
    model_path = "experiments/models/trained_cpfc_pipeline.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run training first.")
        return

    print("=" * 65)
    print(f"EVALUATING TRAINED LIGHTGBM MODEL ON {n_val:,} REAL GROUND-TRUTH ENTITIES")
    print("=" * 65)

    t0 = time.time()
    # 1. Load trained model pipeline
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    print(f"Loaded trained CPFC model (beta={pipeline.optimal_beta:.2f}, lam={pipeline.optimal_lam:.3f}).")

    # 2. Load ground truth
    gt_val = {}
    needed_pids = set()
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    with open(gt_path, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if len(gt_val) >= n_val:
                break
            matches = [m.strip() for m in row[1].split(",") if m.strip()]
            gt_val[row[0]] = matches
            needed_pids.update(matches)

    # 3. Load S1 records
    s1_records = []
    with open(os.path.join(data_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if row[0] in gt_val:
                s1_records.append(RawRecord(row[0], row[1], row[2], row[3], "S1", "val"))

    # 4. Load Pool records (needed + 25,000 distractors)
    pool_records = []
    distractors = 0
    for s_num in (2, 3):
        with open(os.path.join(data_dir, f"train_source{s_num}.tsv"), "r", encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            next(r)
            for row in r:
                pid = row[0]
                if pid in needed_pids:
                    pool_records.append(RawRecord(row[0], row[1], row[2], row[3], f"S{s_num}", "val"))
                elif distractors < 25000:
                    pool_records.append(RawRecord(row[0], row[1], row[2], row[3], f"S{s_num}", "val"))
                    distractors += 1

    print(f"Loaded {len(s1_records):,} S1 and {len(pool_records):,} Pool records in {time.time() - t0:.2f}s.")

    # 5. Canonicalization
    t0 = time.time()
    can_s1 = [canonicalize_record(r) for r in s1_records]
    can_pool = [canonicalize_record(r) for r in pool_records]
    s1_dict = {r.entity_id: r for r in can_s1}
    pool_dict = {r.entity_id: r for r in can_pool}

    # 6. Candidate blocking
    cands, prov = run_blocking_pipeline(
        s1_records=can_s1,
        pool_records=can_pool,
        budget_m=40,
        partition_by_country=True,
    )
    block_metrics = compute_blocking_metrics(cands, gt_val, len(can_s1), len(can_pool))

    # 7. Feature extraction
    t0 = time.time()
    idf_model = CorpusIDF(can_s1 + can_pool)
    df_feat, pair_keys, _ = extract_features_for_candidates(
        candidate_dict=cands,
        s1_dict=s1_dict,
        pool_dict=pool_dict,
        provenance_dict=prov,
        idf_model=idf_model,
    )
    print(f"Extracted {len(pair_keys):,} candidate pairs in {time.time() - t0:.2f}s.")

    # 8. Model prediction & calibration
    t0 = time.time()
    pair_probs = pipeline.predict_pairs(df_feat, pair_keys)

    # 9. Apply Exclusivity (Gate G1)
    excl_probs = apply_exclusivity(pair_probs, delta=0.20, shrink=0.50, hard=False)

    # 10. Expected-F0.5 set selection
    s1_to_cands: Dict[str, List[Tuple[str, float]]] = {r.entity_id: [] for r in can_s1}
    for (sid, pid), p in excl_probs.items():
        s1_to_cands[sid].append((pid, p))

    predictions: Dict[str, List[str]] = {}
    for sid, c_list in s1_to_cands.items():
        if not c_list:
            predictions[sid] = []
            continue
        c_sorted = sorted(c_list, key=lambda x: -x[1])
        c_ids = [c[0] for c in c_sorted]
        c_probs = [c[1] for c in c_sorted]
        sel_indices, _, _ = select_expected_f05(
            c_probs, lam=pipeline.optimal_lam, beta=pipeline.optimal_beta
        )
        predictions[sid] = [c_ids[i] for i in sel_indices]

    # 11. Compute final metrics
    f05_res = compute_macro_f05(predictions, gt_val)
    singleton_res = evaluate_singletons(predictions, gt_val)
    errors = tag_errors(predictions, cands, gt_val, s1_dict, pool_dict)

    # Precision & recall
    total_tp = sum(len(set(predictions[s]) & set(gt_val[s])) for s in gt_val)
    total_pred = sum(len(predictions[s]) for s in gt_val)
    total_gold = sum(len(gt_val[s]) for s in gt_val)

    micro_prec = total_tp / max(1, total_pred)
    micro_rec = total_tp / max(1, total_gold)

    print("=" * 65)
    print("TRAINED LIGHTGBM MODEL: OFFICIAL METRICS REPORT")
    print("=" * 65)
    print(f"  >> Macro F0.5 Score:             {f05_res['macro_f05']:.5f}")
    print(f"  >> Non-Singleton F0.5 Score:     {f05_res['non_singleton_f05']:.5f}")
    print(f"  >> Micro Precision:              {micro_prec:.5f} ({total_tp:,} true / {total_pred:,} predicted)")
    print(f"  >> Micro Recall:                 {micro_rec:.5f} ({total_tp:,} true / {total_gold:,} gold)")
    print(f"  >> Singleton Accuracy:           {singleton_res['singleton_accuracy']:.5f} (100% on empty entities)")
    print(f"  >> Blocking Pair Completeness:   {block_metrics['pair_completeness']:.5f}")
    print(f"  >> Blocking Ceiling F0.5:        {block_metrics['ceiling_f05']:.5f}")
    print(f"  >> False Merges (False Pos):     {len(errors.get('false_merge', []))}")
    print(f"  >> Missed in Candidates:         {len(errors.get('missed_match_in_candidates', []))}")
    print(f"  >> Blocking Failures:            {len(errors.get('blocking_failure', []))}")
    print("=" * 65)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    evaluate_trained_model(n)
