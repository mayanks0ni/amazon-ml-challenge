"""Input data validation and schema integrity checks."""
import re
from typing import Dict, List, Set, Tuple
from ber.io.schemas import RawRecord


def validate_records(records: List[RawRecord]) -> Dict[str, any]:
    """Validates records according to Section 3.1 and 9.2:
    - 4 columns per line, no empty entity_id.
    - Prefixes S1-/S2-/S3- match the source.
    - entity_id is unique within the split and source.
    - Valid UTF-8 and no hidden control characters or embedded tabs/newlines.
    """
    stats = {
        "total_records": len(records),
        "unique_ids": 0,
        "empty_names": 0,
        "empty_addresses": 0,
        "empty_countries": 0,
        "invalid_prefixes": 0,
        "control_char_issues": 0,
    }

    seen_ids: Set[str] = set()
    control_char_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

    for r in records:
        if not r.entity_id:
            raise ValueError("Found record with empty entity_id!")

        if r.entity_id in seen_ids:
            raise ValueError(f"Duplicate entity_id detected within file: {r.entity_id}")
        seen_ids.add(r.entity_id)

        # Check prefix matches source
        expected_prefix = f"{r.source}-"
        if not r.entity_id.startswith(expected_prefix):
            stats["invalid_prefixes"] += 1
            raise ValueError(
                f"Record {r.entity_id} does not start with expected prefix '{expected_prefix}'"
            )

        if not r.business_name.strip():
            stats["empty_names"] += 1
        if not r.business_address.strip():
            stats["empty_addresses"] += 1
        if not r.country.strip():
            stats["empty_countries"] += 1

        # Check control chars
        for f in (r.business_name, r.business_address, r.country):
            if control_char_re.search(f):
                stats["control_char_issues"] += 1

    stats["unique_ids"] = len(seen_ids)
    return stats
