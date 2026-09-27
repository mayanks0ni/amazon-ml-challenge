"""Command-Line Interface for Business Entity Resolution CPFC pipeline."""
import argparse
import json
import os
import sys
import yaml
import numpy as np
import pandas as pd
from ber.data.synthetic import generate_synthetic_benchmark
from ber.io.read_tsv import load_source_records, load_ground_truth
from ber.validation.input_checks import validate_records
from ber.validation.gt_checks import evaluate_decision_gates
from ber.normalization.pipeline import canonicalize_record
from ber.blocking import run_blocking_pipeline, compute_blocking_metrics
from ber.features.registry import (
    extract_features_for_candidates,
    get_monotone_constraints,
)
from ber.features.rarity import CorpusIDF
from ber.training.oof import run_oof_training_pipeline
from ber.inference.predict import run_test_inference
from ber.evaluation.report import generate_experiment_report


def run_pipeline(config: dict) -> None:
    """Executes complete end-to-end training and test inference pipeline."""
    print("=" * 60)
    print("Business Entity Resolution: Calibrated Precision-First Cascade (CPFC)")
    print("=" * 60)

    train_dir = config.get("data", {}).get("train_dir", "dataset/train")
    test_dir = config.get("data", {}).get("test_dir", "dataset/test")
    output_dir = config.get("data", {}).get("output_dir", "output")
    exp_dir = config.get("experiments", {}).get("dir", "experiments")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    # 1. Ingest and Validate Train Data
    print("\n[Phase 1] Ingesting and validating training records...")
    s1_train = load_source_records(os.path.join(train_dir, "train_source1.tsv"), "S1", "train")
    s2_train = load_source_records(os.path.join(train_dir, "train_source2.tsv"), "S2", "train")
    s3_train = load_source_records(os.path.join(train_dir, "train_source3.tsv"), "S3", "train")
    gt = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))

    for name, recs in [("S1", s1_train), ("S2", s2_train), ("S3", s3_train)]:
        stats = validate_records(recs)
        print(f"  {name} valid: {stats['unique_ids']} unique records.")

    # 2. Evaluate Decision Gates G1-G6
    print("\n[Phase 2] Evaluating EDA Decision Gates G1-G6...")
    gate_results = evaluate_decision_gates(s1_train, s2_train, s3_train, gt)
    print(f"  Gate G1 (Hard Exclusivity): {gate_results['G1_hard_exclusivity']}")
    print(f"  Gate G2 (Country Partitioning): {gate_results['G2_partition_by_country']} (same-country rate: {gate_results['G2_country_rate']:.4f})")
    print(f"  Gate G3 (Numeric Primary): {gate_results['G3_numeric_primary']} (postal presence: {gate_results['G3_postal_rate']:.4f})")
    print(f"  Gate G6 (Stage 2b Priority): {gate_results['G6_stage2b_priority']} (multi-match share: {gate_results['G6_multi_match_share']:.4f})")
    print(f"  Singleton rate: {gate_results['singleton_rate']:.4f} ({gate_results['total_singletons']}/{gate_results['total_s1']})")

    # 3. Canonical Normalization
    print("\n[Phase 3] Canonicalizing training records...")
    can_s1 = [canonicalize_record(r) for r in s1_train]
    can_pool = [canonicalize_record(r) for r in (s2_train + s3_train)]

    s1_dict = {r.entity_id: r for r in can_s1}
    pool_dict = {r.entity_id: r for r in can_pool}

    # 4. Multi-Channel Candidate Blocking
    budget_m = config.get("blocking", {}).get("budget_m", 60)
    print(f"\n[Phase 4] Multi-channel blocking (Budget M={budget_m})...")
    candidates_train, cand_prov_train = run_blocking_pipeline(
        s1_records=can_s1,
        pool_records=can_pool,
        budget_m=budget_m,
        partition_by_country=gate_results["G2_partition_by_country"],
    )

    block_metrics = compute_blocking_metrics(
        candidates_train, gt, num_s1=len(can_s1), num_pool=len(can_pool)
    )
    print(f"  Pair Completeness: {block_metrics['pair_completeness']:.4f}")
    print(f"  Entity Recall:     {block_metrics['entity_recall']:.4f}")
    print(f"  Ceiling F0.5:      {block_metrics['ceiling_f05']:.4f}")
    print(f"  Reduction Ratio:   {block_metrics['reduction_ratio']:.6f}")
    print(f"  Mean Candidates:   {block_metrics['mean_candidates_per_s1']:.1f}")

    # 5. Feature Engineering
    print("\n[Phase 5] Extracting ~150 features on blocked pairs...")
    idf_model = CorpusIDF(can_s1 + can_pool)
    df_features_2a, pair_keys, feat_cols = extract_features_for_candidates(
        candidate_dict=candidates_train,
        s1_dict=s1_dict,
        pool_dict=pool_dict,
        provenance_dict=cand_prov_train,
        idf_model=idf_model,
    )
    print(f"  Generated {df_features_2a.shape[0]} pairs with {df_features_2a.shape[1]} features.")

    # Construct binary labels (1 iff pool_id in gt[s1_id])
    labels = np.array(
        [1 if pair[1] in gt.get(pair[0], []) else 0 for pair in pair_keys],
        dtype=np.int32,
    )
    print(f"  Label balance: {np.sum(labels)} positives / {len(labels) - np.sum(labels)} negatives ({np.mean(labels)*100:.2f}% pos).")

    # 6. Model Training & Calibration
    print("\n[Phase 6] 5-fold GroupKFold training (Stage 2a + Stage 2b + Isotonic)...")
    monotone = get_monotone_constraints(feat_cols)
    model_pipeline, oof_results = run_oof_training_pipeline(
        df_features_2a=df_features_2a,
        pair_keys=pair_keys,
        labels=labels,
        monotone_constraints=monotone,
        ground_truth=gt,
        n_splits=config.get("training", {}).get("n_splits", 5),
        seed=config.get("training", {}).get("seed", 42),
    )

    print(f"\n[Phase 7] Validation Performance (Out-of-Fold):")
    print(f"  >> Macro F0.5:         {oof_results['macro_f05']:.4f}")
    print(f"  >> Singleton Accuracy: {oof_results['singleton_acc']:.4f}")
    print(f"  >> Non-Singleton F0.5: {oof_results['non_singleton_f05']:.4f}")
    print(f"  >> Calibrated ECE:     {oof_results.get('calibrated_ece', 0.0):.4f}")

    # Save metrics report
    report_path = os.path.join(exp_dir, "metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(oof_results, f, indent=2)
    print(f"  Saved validation report to: {report_path}")

    # 7. Test Inference & Submission Generation
    if os.path.exists(os.path.join(test_dir, "test_source1.tsv")):
        print(f"\n[Phase 8] Running test inference on {test_dir}...")
        match_path, cand_path = run_test_inference(
            test_dir=test_dir,
            model_pipeline=model_pipeline,
            output_dir=output_dir,
            budget_m=budget_m,
        )
        print(f"  Wrote candidate pairs: {cand_path}")
        print(f"  Wrote matching results: {match_path}")

        # Run Validator
        print("\n[Phase 9] Running submission validator...")
        from utils.validate_submission import main as run_validator_main
        # Test directly via subprocess
        import subprocess
        v_res = subprocess.run(
            [
                sys.executable,
                "utils/validate_submission.py",
                "--matching",
                match_path,
                "--candidate",
                cand_path,
                "--test-dir",
                test_dir,
            ],
            capture_output=True,
            text=True,
        )
        print(v_res.stdout.strip())
        if v_res.returncode != 0:
            print(v_res.stderr.strip())
            sys.exit(1)

    print("\nCPFC Pipeline completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution CLI.")
    subparsers = parser.add_subparsers(dest="command")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run full CPFC pipeline.")
    run_parser.add_argument("--config", default="configs/final.yaml", help="Path to config YAML.")

    # Command: generate-synthetic
    gen_parser = subparsers.add_parser("generate-synthetic", help="Generate synthetic benchmark data.")
    gen_parser.add_argument("--base-dir", default="dataset", help="Output directory for datasets.")
    gen_parser.add_argument("--train-size", type=int, default=100, help="Number of train S1 records.")
    gen_parser.add_argument("--test-size", type=int, default=50, help="Number of test S1 records.")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Command: validate
    val_parser = subparsers.add_parser("validate", help="Validate submission files.")
    val_parser.add_argument("--matching", default="output/matching_results.tsv")
    val_parser.add_argument("--candidate", default="output/candidate_pairs.tsv")
    val_parser.add_argument("--test-dir", default="dataset/test")

    args = parser.parse_args()

    if args.command == "run":
        if os.path.exists(args.config):
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {}
        run_pipeline(cfg)

    elif args.command == "generate-synthetic":
        print(f"Generating synthetic benchmark (train: {args.train_size}, test: {args.test_size}) in {args.base_dir}...")
        generate_synthetic_benchmark(
            base_dir=args.base_dir,
            train_size=args.train_size,
            test_size=args.test_size,
            seed=args.seed,
        )
        print("Dataset generated successfully!")

    elif args.command == "validate":
        import subprocess
        v_res = subprocess.run(
            [
                sys.executable,
                "utils/validate_submission.py",
                "--matching",
                args.matching,
                "--candidate",
                args.candidate,
                "--test-dir",
                args.test_dir,
            ],
            capture_output=True,
            text=True,
        )
        print(v_res.stdout.strip())
        sys.exit(v_res.returncode)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
