"""End-to-end training, validation, and benchmarking on real competition dataset."""
import csv
import json
import os
import random
import time
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

from ber.io.schemas import RawRecord, CanonicalRecord
from ber.normalization.pipeline import canonicalize_record
from ber.blocking import run_blocking_pipeline, compute_blocking_metrics
from ber.features.registry import extract_features_for_candidates, get_monotone_constraints
from ber.features.rarity import CorpusIDF
from ber.training.oof import run_oof_training_pipeline
from ber.evaluation.f05 import compute_macro_f05
from ber.evaluation.error_tags import tag_errors
from ber.decision.singleton import evaluate_singletons
from ber.decision.exclusivity import apply_exclusivity
from ber.decision.expected_f import select_expected_f05


def extract_real_cohort(
    data_dir: str = "student_resource/dataset/train",
    n_train_s1: int = 10000,
    n_val_s1: int = 2500,
    n_distractors: int = 40000,
    seed: int = 42,
    full_dataset: bool = False,
) -> Tuple[
    List[RawRecord], List[RawRecord], Dict[str, List[str]],
    List[RawRecord], List[RawRecord], Dict[str, List[str]],
]:
    """Extracts train and validation cohorts with true matches and distractors.
    Supports complete dataset training via full_dataset=True.
    """
    if full_dataset:
        print(f"Loading COMPLETE dataset from {data_dir} (all 2.2M S1 records)...")
    else:
        print(f"Sampling {n_train_s1} train S1 and {n_val_s1} val S1 from {data_dir}...")
    rng = random.Random(seed)

    gt_train_map: Dict[str, List[str]] = {}
    gt_val_map: Dict[str, List[str]] = {}
    needed_train_pool_ids: Set[str] = set()
    needed_val_pool_ids: Set[str] = set()

    # Step 1: Read ground truth
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            sid = row[0]
            matches = [m.strip() for m in row[1].split(",") if m.strip()]
            if len(gt_val_map) < n_val_s1:
                gt_val_map[sid] = matches
                needed_val_pool_ids.update(matches)
            elif full_dataset or len(gt_train_map) < n_train_s1:
                gt_train_map[sid] = matches
                needed_train_pool_ids.update(matches)
            else:
                break

    all_target_s1 = set(gt_train_map.keys()).union(set(gt_val_map.keys()))

    # Step 2: Read S1 records
    s1_train_records: List[RawRecord] = []
    s1_val_records: List[RawRecord] = []
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            sid = row[0]
            if sid in gt_train_map:
                s1_train_records.append(RawRecord(row[0], row[1], row[2], row[3], "S1", "train"))
            elif sid in gt_val_map:
                s1_val_records.append(RawRecord(row[0], row[1], row[2], row[3], "S1", "train"))

    # Step 3: Read Pool records (S2 and S3)
    pool_train_records: List[RawRecord] = []
    pool_val_records: List[RawRecord] = []
    distractors: List[RawRecord] = []

    for s_num in (2, 3):
        p_path = os.path.join(data_dir, f"train_source{s_num}.tsv")
        with open(p_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                pid = row[0]
                rec = RawRecord(row[0], row[1], row[2], row[3], f"S{s_num}", "train")
                if pid in needed_train_pool_ids:
                    pool_train_records.append(rec)
                elif pid in needed_val_pool_ids:
                    pool_val_records.append(rec)
                elif full_dataset:
                    pool_train_records.append(rec)
                elif len(distractors) < n_distractors:
                    distractors.append(rec)

    # Distribute distractors across train and val pools if sampling
    if not full_dataset and distractors:
        half_dist = len(distractors) // 2
        pool_train_records.extend(distractors[:half_dist])
        pool_val_records.extend(distractors[half_dist:])

    print(f"Extracted Train: {len(s1_train_records):,} S1, {len(pool_train_records):,} Pool records.")
    print(f"Extracted Val:   {len(s1_val_records):,} S1, {len(pool_val_records):,} Pool records.")

    return (
        s1_train_records, pool_train_records, gt_train_map,
        s1_val_records, pool_val_records, gt_val_map,
    )


def train_and_evaluate_real(
    data_dir: str = "student_resource/dataset/train",
    n_train_s1: int = 10000,
    n_val_s1: int = 2500,
    n_distractors: int = 40000,
    budget_m: int = 50,
    full_dataset: bool = False,
    seed: int = 42,
) -> dict:
    """Executes training and validation benchmarking on the real dataset."""
    t_start = time.time()
    print("=" * 70)
    mode_str = "COMPLETE DATASET" if full_dataset else f"{n_train_s1:,} COHORT"
    print(f"CPFC ON REAL COMPETITION DATASET: TRAINING & BENCHMARKING ({mode_str})")
    print("=" * 70)

    # 1. Extract real cohort
    (
        s1_train, pool_train, gt_train,
        s1_val, pool_val, gt_val,
    ) = extract_real_cohort(
        data_dir=data_dir,
        n_train_s1=n_train_s1,
        n_val_s1=n_val_s1,
        n_distractors=n_distractors,
        seed=seed,
    )

    # 2. Canonicalization
    t0 = time.time()
    print("\nCanonicalizing training and validation records...")
    can_s1_train = [canonicalize_record(r) for r in s1_train]
    can_pool_train = [canonicalize_record(r) for r in pool_train]
    can_s1_val = [canonicalize_record(r) for r in s1_val]
    can_pool_val = [canonicalize_record(r) for r in pool_val]
    print(f"Canonicalization completed in {time.time() - t0:.2f}s.")

    s1_train_dict = {r.entity_id: r for r in can_s1_train}
    pool_train_dict = {r.entity_id: r for r in can_pool_train}
    s1_val_dict = {r.entity_id: r for r in can_s1_val}
    pool_val_dict = {r.entity_id: r for r in can_pool_val}

    # 3. Multi-Channel Candidate Blocking
    t0 = time.time()
    print(f"\nRunning multi-channel blocking on training cohort (Budget M={budget_m})...")
    cands_train, prov_train = run_blocking_pipeline(
        s1_records=can_s1_train,
        pool_records=can_pool_train,
        budget_m=budget_m,
        partition_by_country=True,
    )
    block_train_metrics = compute_blocking_metrics(
        cands_train, gt_train, num_s1=len(can_s1_train), num_pool=len(can_pool_train)
    )
    print(f"  Train Pair Completeness: {block_train_metrics['pair_completeness']:.4f}")
    print(f"  Train Entity Recall:     {block_train_metrics['entity_recall']:.4f}")
    print(f"  Train Ceiling F0.5:      {block_train_metrics['ceiling_f05']:.4f}")
    print(f"  Train Reduction Ratio:   {block_train_metrics['reduction_ratio']:.6f}")
    print(f"  Mean Candidates per S1:  {block_train_metrics['mean_candidates_per_s1']:.1f}")

    # 4. Feature Extraction
    t0 = time.time()
    print("\nExtracting ~150 features on training pairs...")
    idf_model = CorpusIDF(can_s1_train + can_pool_train)
    df_feat_train, pair_keys_train, feat_cols = extract_features_for_candidates(
        candidate_dict=cands_train,
        s1_dict=s1_train_dict,
        pool_dict=pool_train_dict,
        provenance_dict=prov_train,
        idf_model=idf_model,
    )
    labels_train = np.array(
        [1 if p[1] in gt_train.get(p[0], []) else 0 for p in pair_keys_train],
        dtype=np.int32,
    )
    print(f"Extracted {df_feat_train.shape[0]:,} training pairs with {df_feat_train.shape[1]} features in {time.time() - t0:.2f}s.")
    print(f"Label distribution: {np.sum(labels_train):,} positives ({np.mean(labels_train)*100:.2f}%), {len(labels_train)-np.sum(labels_train):,} negatives.")

    # 5. Model Training (Stage 2a + Stage 2b + Isotonic)
    t0 = time.time()
    print("\nTraining 5-fold LightGBM (monotone constraints, GroupKFold by S1)...")
    monotone = get_monotone_constraints(feat_cols)
    model_pipeline, oof_results = run_oof_training_pipeline(
        df_features_2a=df_feat_train,
        pair_keys=pair_keys_train,
        labels=labels_train,
        monotone_constraints=monotone,
        ground_truth=gt_train,
        n_splits=5,
        seed=42,
    )
    print(f"5-Fold training and calibration completed in {time.time() - t0:.2f}s.")
    print(f"  OOF Macro F0.5:         {oof_results['macro_f05']:.4f}")
    print(f"  OOF Singleton Accuracy: {oof_results['singleton_acc']:.4f}")
    print(f"  OOF Calibrated ECE:     {oof_results.get('calibrated_ece', 0.0):.6f}")
    print(f"  Optimal Decision Beta:  {model_pipeline.optimal_beta:.2f}")

    # 6. Evaluation on Held-Out Validation Cohort
    t0 = time.time()
    print("\nEvaluating on held-out validation cohort (2,000 S1 entities)...")
    cands_val, prov_val = run_blocking_pipeline(
        s1_records=can_s1_val,
        pool_records=can_pool_val,
        budget_m=budget_m,
        partition_by_country=True,
    )
    block_val_metrics = compute_blocking_metrics(
        cands_val, gt_val, num_s1=len(can_s1_val), num_pool=len(can_pool_val)
    )

    df_feat_val, pair_keys_val, _ = extract_features_for_candidates(
        candidate_dict=cands_val,
        s1_dict=s1_val_dict,
        pool_dict=pool_val_dict,
        provenance_dict=prov_val,
        idf_model=idf_model,
    )

    # Score validation pairs
    val_pair_probs = model_pipeline.predict_pairs(df_feat_val, pair_keys_val)

    # Apply Exclusivity (Gate G1)
    val_excl_probs = apply_exclusivity(val_pair_probs, delta=0.10, shrink=0.25, hard=False)

    # Expected-F0.5 set selection
    val_s1_to_cands: Dict[str, List[Tuple[str, float]]] = {r.entity_id: [] for r in can_s1_val}
    for (sid, pid), p in val_excl_probs.items():
        val_s1_to_cands[sid].append((pid, p))

    val_predictions: Dict[str, List[str]] = {}
    for sid in val_s1_to_cands:
        cands = val_s1_to_cands[sid]
        if not cands:
            val_predictions[sid] = []
            continue
        cands_sorted = sorted(cands, key=lambda x: -x[1])
        c_ids = [c[0] for c in cands_sorted]
        c_probs = [c[1] for c in cands_sorted]
        sel_indices, k, v = select_expected_f05(
            c_probs, lam=model_pipeline.optimal_lam, beta=model_pipeline.optimal_beta
        )
        val_predictions[sid] = [c_ids[i] for i in sel_indices]

    val_f05_res = compute_macro_f05(val_predictions, gt_val)
    val_singleton_res = evaluate_singletons(val_predictions, gt_val)
    val_errors = tag_errors(val_predictions, cands_val, gt_val, s1_val_dict, pool_val_dict)

    # Print Report
    print("=" * 70)
    print("FINAL VALIDATION REPORT (HELD-OUT REAL COHORT)")
    print("=" * 70)
    print(f"  >> Macro F0.5:                 {val_f05_res['macro_f05']:.4f}")
    print(f"  >> Non-Singleton F0.5:         {val_f05_res['non_singleton_f05']:.4f}")
    print(f"  >> Singleton Accuracy:         {val_singleton_res['singleton_accuracy']:.4f}")
    print(f"  >> Non-Singleton Abstention:   {val_singleton_res['non_singleton_abstention_rate']:.4f}")
    print(f"  >> Blocking Pair Completeness: {block_val_metrics['pair_completeness']:.4f}")
    print(f"  >> Blocking Ceiling F0.5:      {block_val_metrics['ceiling_f05']:.4f}")
    print(f"  >> False Merges:               {len(val_errors.get('false_merge', []))}")
    print(f"  >> Missed Matches in Cands:    {len(val_errors.get('missed_match_in_candidates', []))}")
    print(f"  >> Blocking Failures:          {len(val_errors.get('blocking_failure', []))}")
    print(f"  >> Total Validation Time:      {time.time() - t_start:.2f}s")
    print("=" * 70)

    # Save metrics to json
    results = {
        "val_macro_f05": val_f05_res["macro_f05"],
        "val_non_singleton_f05": val_f05_res["non_singleton_f05"],
        "val_singleton_acc": val_singleton_res["singleton_accuracy"],
        "val_blocking_pc": block_val_metrics["pair_completeness"],
        "val_blocking_ceiling": block_val_metrics["ceiling_f05"],
        "optimal_beta": model_pipeline.optimal_beta,
    }
    with open("experiments/real_benchmark_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    import pickle
    os.makedirs("experiments/models", exist_ok=True)
    model_save_path = "experiments/models/trained_cpfc_pipeline.pkl"
    with open(model_save_path, "wb") as f:
        pickle.dump(model_pipeline, f)
    print(f"Saved trained CPFC pipeline model to {model_save_path}")

    return results, model_pipeline


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CPFC model on real dataset")
    parser.add_argument("--data-dir", default="student_resource/dataset/train", type=str)
    parser.add_argument("--train-s1", default=10000, type=int, help="Number of train S1 entities")
    parser.add_argument("--val-s1", default=2500, type=int, help="Number of validation S1 entities")
    parser.add_argument("--distractors", default=40000, type=int, help="Number of negative distractors")
    parser.add_argument("--budget-m", default=40, type=int, help="Candidate blocking budget per S1")
    parser.add_argument("--full", action="store_true", help="Train on the complete 2.2M dataset")
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    train_and_evaluate_real(
        data_dir=args.data_dir,
        n_train_s1=args.train_s1,
        n_val_s1=args.val_s1,
        n_distractors=args.distractors,
        budget_m=args.budget_m,
        full_dataset=args.full,
        seed=args.seed,
    )
