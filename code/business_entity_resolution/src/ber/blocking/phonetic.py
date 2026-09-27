"""Channel C4: Phonetic folding and acronym index blocking."""
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple
from ber.io.schemas import CanonicalRecord


def retrieve_c4_phonetic_acronym(
    s1_records: Sequence[CanonicalRecord],
    pool_records: Sequence[CanonicalRecord],
    top_k: int = 15,
) -> List[Tuple[str, str, int, float]]:
    """C4: Phonetic / acronym matching.
    Matches:
    - name_acronym <-> name_acronym
    - name_fold rare tokens
    """
    acronym_index: Dict[str, List[str]] = defaultdict(list)
    fold_token_index: Dict[str, List[str]] = defaultdict(list)

    for rec in pool_records:
        if rec.name_acronym and len(rec.name_acronym) >= 2:
            acronym_index[rec.name_acronym].append(rec.entity_id)

        fold_tokens = [t for t in rec.name_fold.split() if len(t) >= 4]
        for t in fold_tokens:
            fold_token_index[t].append(rec.entity_id)

    results: List[Tuple[str, str, int, float]] = []

    for s1 in s1_records:
        matches: Dict[str, float] = defaultdict(float)

        # Acronym match (strong signal)
        if s1.name_acronym and len(s1.name_acronym) >= 2:
            for pid in acronym_index.get(s1.name_acronym, []):
                matches[pid] += 2.0

        # Phonetic token match
        s1_fold_tokens = [t for t in s1.name_fold.split() if len(t) >= 4]
        for t in s1_fold_tokens:
            postings = fold_token_index.get(t, [])
            if 0 < len(postings) <= 30:  # Rare phonetic tokens only
                for pid in postings:
                    matches[pid] += 1.0

        if not matches:
            continue

        sorted_matches = sorted(matches.items(), key=lambda x: -x[1])[:top_k]
        for rank, (pid, score) in enumerate(sorted_matches):
            results.append((s1.entity_id, pid, rank, score))

    return results
