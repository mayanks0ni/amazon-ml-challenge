"""Unit tests for I/O functions and submission writer."""
import os
import tempfile
import pytest
from ber.io.read_tsv import read_tsv
from ber.io.write_submission import write_lists, write_submission_outputs


def test_read_tsv_tricky_values():
    content = "entity_id\tbusiness_name\tbusiness_address\tcountry\n" \
              "S1-1\tNA\tNone\tNull\n" \
              "S1-2\t\"Quoted Business\"\t123 Main St\tUSA\n"
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as f:
        f.write(content)
        f.flush()
        temp_path = f.name

    try:
        df = read_tsv(temp_path, expected_cols=4)
        assert len(df) == 2
        # Ensure 'NA', 'None', 'Null' were NOT converted to NaN
        assert df.iloc[0]["business_name"] == "NA"
        assert df.iloc[0]["business_address"] == "None"
        assert df.iloc[0]["country"] == "Null"
        # Ensure quotes were preserved cleanly without breaking columns
        assert "Quoted Business" in df.iloc[1]["business_name"]
    finally:
        os.remove(temp_path)


def test_write_submission_contract():
    with tempfile.TemporaryDirectory() as tmpdir:
        s1_ids = ["S1-001", "S1-002", "S1-003"]
        cands = {
            "S1-001": ["S2-10", "S3-20"],
            "S1-002": ["S2-30"],
            "S1-003": [],  # singleton
        }
        matches = {
            "S1-001": ["S2-10"],
            "S1-002": ["S2-30"],
            "S1-003": [],  # singleton
        }
        valid_pool = {"S2-10", "S3-20", "S2-30"}

        m_path, c_path = write_submission_outputs(
            test_s1_ids=s1_ids,
            candidates=cands,
            matches=matches,
            valid_pool_ids=valid_pool,
            output_dir=tmpdir,
        )

        assert os.path.exists(m_path)
        assert os.path.exists(c_path)

        # Inspect lines in matching_results.tsv
        with open(m_path, "r", encoding="utf-8") as f:
            lines = [l.rstrip("\r\n") for l in f.readlines()]
        assert lines[0] == "source1_entity_id\tmatched_entity_ids"
        assert lines[1] == "S1-001\tS2-10"
        assert lines[2] == "S1-002\tS2-30"
        assert lines[3] == "S1-003\t"  # Empty string for singleton, tab-separated
