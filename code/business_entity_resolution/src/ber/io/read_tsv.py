"""Safe TSV reader adhering to all competition constraints and edge cases."""
import csv
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
from ber.io.schemas import RawRecord


def read_tsv(path: str, expected_cols: Optional[int] = None) -> pd.DataFrame:
    """Reads TSV file with strict settings preventing NaN coercion and quote errors.

    Guarantees:
    - Business names like 'NA', 'None', 'Null' stay strings.
    - Quotes do not swallow tabs or newlines.
    - Explicit tab delimiter.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"TSV file not found: {path}")

    # Preliminary structural verification
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().rstrip("\r\n")
        header_cols = header_line.split("\t")
        if expected_cols is not None and len(header_cols) != expected_cols:
            raise ValueError(
                f"File {path} header has {len(header_cols)} columns, expected {expected_cols}: {header_cols}"
            )

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )

    first_col = df.columns.tolist()[0]
    if first_col not in ("entity_id", "source1_entity_id"):
        raise ValueError(
            f"Invalid first column in {path}: {first_col}. Expected 'entity_id' or 'source1_entity_id'."
        )

    return df


def load_source_records(
    path: str,
    source_name: str,
    split: str = "train",
) -> List[RawRecord]:
    """Loads a source file (train_source1.tsv, etc.) into a list of RawRecord objects."""
    df = read_tsv(path, expected_cols=4)
    expected_headers = ["entity_id", "business_name", "business_address", "country"]
    if df.columns.tolist() != expected_headers:
        raise ValueError(f"Unexpected headers in {path}: {df.columns.tolist()} vs {expected_headers}")

    records: List[RawRecord] = []
    for _, row in df.iterrows():
        records.append(
            RawRecord(
                entity_id=row["entity_id"],
                business_name=row["business_name"],
                business_address=row["business_address"],
                country=row["country"],
                source=source_name,
                split=split,
            )
        )
    return records


def load_ground_truth(path: str) -> Dict[str, List[str]]:
    """Loads train_ground_truth.tsv into a dictionary of source1_entity_id -> list of matched IDs."""
    df = read_tsv(path, expected_cols=2)
    expected_headers = ["source1_entity_id", "matched_entity_ids"]
    if df.columns.tolist() != expected_headers:
        raise ValueError(f"Unexpected ground truth headers in {path}: {df.columns.tolist()}")

    gt: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        sid = row["source1_entity_id"]
        raw_matches = row["matched_entity_ids"]
        # Split on comma, stripping empty elements
        if raw_matches:
            matches = [m.strip() for m in raw_matches.split(",") if m.strip()]
        else:
            matches = []
        gt[sid] = matches
    return gt
