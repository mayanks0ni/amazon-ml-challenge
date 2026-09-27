"""Contract and submission format integration tests (Section 27.2)."""
import os
import subprocess
import sys
import tempfile
import pytest
from ber.data.synthetic import generate_synthetic_benchmark
from ber.io.write_submission import write_submission_outputs


def test_submission_contract_and_validator():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "dataset")
        out_dir = os.path.join(tmpdir, "output")

        # 1. Generate small synthetic benchmark
        generate_synthetic_benchmark(base_dir=data_dir, train_size=20, test_size=15, seed=42)

        # 2. Load actual generated pool IDs
        from ber.io.read_tsv import load_source_records
        test_s1_records = load_source_records(os.path.join(data_dir, "test", "test_source1.tsv"), "S1", "test")
        test_s2_records = load_source_records(os.path.join(data_dir, "test", "test_source2.tsv"), "S2", "test")
        test_s3_records = load_source_records(os.path.join(data_dir, "test", "test_source3.tsv"), "S3", "test")

        test_s1_ids = [r.entity_id for r in test_s1_records]
        test_pool_ids = {r.entity_id for r in (test_s2_records + test_s3_records)}
        pool_id_list = sorted(list(test_pool_ids))

        # Assign first 2 available pool records as candidates
        candidates = {
            sid: pool_id_list[:2] for sid in test_s1_ids
        }
        # First half get a match, second half are singletons
        matches = {
            sid: ([pool_id_list[0]] if idx < (len(test_s1_ids) // 2) else [])
            for idx, sid in enumerate(test_s1_ids)
        }

        match_path, cand_path = write_submission_outputs(
            test_s1_ids=test_s1_ids,
            candidates=candidates,
            matches=matches,
            valid_pool_ids=test_pool_ids,
            output_dir=out_dir,
        )

        # 3. Execute validate_submission.py
        possible_paths = [
            os.path.abspath("utils/validate_submission.py"),
            os.path.abspath("../../utils/validate_submission.py"),
            os.path.abspath("../student_resource/utils/validate_submission.py"),
        ]
        validator_script = next((p for p in possible_paths if os.path.exists(p)), possible_paths[0])
        cmd = [
            sys.executable,
            validator_script,
            "--matching",
            match_path,
            "--candidate",
            cand_path,
            "--test-dir",
            os.path.join(data_dir, "test"),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Validator failed: {result.stdout}\n{result.stderr}"
        assert "PASS" in result.stdout and "Safe to submit" in result.stdout
