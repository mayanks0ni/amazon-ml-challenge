"""DBA, trade name, and alias extraction (Section 10.1)."""
import re
from typing import List, Tuple

# Patterns identifying trade names / DBAs
DBA_PATTERNS = [
    r"\b(?:d/?b/?a|doing\s+business\s+as)\b\s*(.+)",
    r"\b(?:t/?a|trading\s+as)\b\s*(.+)",
    r"\b(?:a/?k/?a|also\s+known\s+as)\b\s*(.+)",
    r"\(([^)]+)\)",  # Parenthetical alternate names like "X (Y Mart)"
]


def extract_dba_aliases(name_norm: str) -> Tuple[str, List[str]]:
    """Splits normalized name into primary name and alternative trade names/aliases."""
    if not name_norm:
        return "", []

    primary = name_norm.lower()
    alts: List[str] = []

    # Check parenthetical names first: "company a (shop b)"
    paren_matches = re.findall(r"\(([^)]+)\)", primary)
    for p in paren_matches:
        p_clean = p.strip()
        if len(p_clean) >= 2:
            alts.append(p_clean)
    primary = re.sub(r"\([^)]*\)", " ", primary)

    # Check dba / t/a / aka
    for pattern in DBA_PATTERNS[:3]:
        m = re.search(pattern, primary)
        if m:
            alt_name = m.group(1).strip()
            if alt_name and len(alt_name) >= 2:
                alts.append(alt_name)
            # Remove dba suffix from primary
            primary = primary[: m.start()].strip()

    primary = re.sub(r"\s+", " ", primary).strip(" ,.-/")
    # Clean alts
    cleaned_alts = [re.sub(r"\s+", " ", a).strip(" ,.-/") for a in alts]
    cleaned_alts = [a for a in cleaned_alts if a and a != primary]

    return primary, cleaned_alts
