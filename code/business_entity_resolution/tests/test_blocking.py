"""Unit tests for multi-channel blocking pipeline and pruning."""
import pytest
from ber.io.schemas import RawRecord
from ber.normalization.pipeline import canonicalize_record
from ber.blocking import run_blocking_pipeline, compute_blocking_metrics


def test_blocking_multi_channel():
    # Construct synthetic reference and dirty pool records
    s1_raw = [
        RawRecord("S1-1", "Infosys Technologies Ltd", "Hosur Road, Electronics City, Bangalore 560100", "India", "S1"),
        RawRecord("S1-2", "Starbucks Coffee", "123 Main St, Seattle, WA 98101", "US", "S1"),
        RawRecord("S1-3", "Acme Corporation", "10 Elm Ave, Austin, TX 78701", "US", "S1"),
    ]
    s2_raw = [
        RawRecord("S2-10", "Infosys Ltd", "Electronics City, Bangalore 560100", "India", "S2"),
        RawRecord("S2-20", "Starbucks", "123 Main Street, Seattle 98101", "US", "S2"),
    ]
    s3_raw = [
        RawRecord("S3-100", "Infosys Tech", "Bangalore 560100", "India", "S3"),
        RawRecord("S3-200", "Distractor Corp", "999 Other St, Dallas, TX 75001", "US", "S3"),
    ]

    s1_can = [canonicalize_record(r) for r in s1_raw]
    pool_can = [canonicalize_record(r) for r in (s2_raw + s3_raw)]

    cands, prov = run_blocking_pipeline(
        s1_can, pool_can, budget_m=10, enable_s2_s3_expansion=True
    )

    # Verify S1-1 retrieved S2-10 and S3-100
    assert "S2-10" in cands["S1-1"]
    # Verify S1-2 retrieved S2-20
    assert "S2-20" in cands["S1-2"]
    # Verify all S1 records are present in candidate dict
    assert "S1-3" in cands

    # Verify metrics
    gt = {
        "S1-1": ["S2-10", "S3-100"],
        "S1-2": ["S2-20"],
        "S1-3": [],  # singleton
    }
    metrics = compute_blocking_metrics(cands, gt, num_s1=3, num_pool=4)
    assert metrics["pair_completeness"] == 1.0
    assert metrics["entity_recall"] == 1.0
    assert metrics["ceiling_f05"] == 1.0
