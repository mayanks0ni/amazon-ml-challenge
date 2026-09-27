"""Channel C3: Rare-token inverted index blocking (Section 12.3)."""
from collections import Counter, defaultdict
import math
from typing import Dict, List, Sequence, Set, Tuple
from ber.io.schemas import CanonicalRecord


def retrieve_c3_rare_tokens(
    s1_records: Sequence[CanonicalRecord],
    pool_records: Sequence[CanonicalRecord],
    max_df: int = 200,
    top_k: int = 50,
) -> List[Tuple[str, str, int, float]]:
    """C3: Inverted index on rare core tokens (df <= max_df).
    Scores candidate pairs by sum of IDF of shared rare tokens.
    """
    total_docs = len(pool_records) + len(s1_records)
    if total_docs == 0:
        return []

    # Count document frequency of tokens in pool
    df_counter: Counter = Counter()
    inverted_index: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    for rec in pool_records:
        unique_tokens = set(rec.name_core_tokens)
        for t in unique_tokens:
            if len(t) >= 2 or t.isdigit():
                df_counter[t] += 1

    # Build inverted index only for tokens with df <= max_df
    for rec in pool_records:
        unique_tokens = set(rec.name_core_tokens)
        for t in unique_tokens:
            if (len(t) >= 2 or t.isdigit()) and df_counter[t] <= max_df:
                idf = math.log((total_docs + 1.0) / (df_counter[t] + 1.0)) + 1.0
                inverted_index[t].append((rec.entity_id, idf))

    results: List[Tuple[str, str, int, float]] = []

    for s1 in s1_records:
        candidate_scores: Dict[str, float] = defaultdict(float)
        unique_s1_tokens = set(s1.name_core_tokens)

        for t in unique_s1_tokens:
            if (len(t) >= 2 or t.isdigit()) and t in inverted_index:
                for pool_id, idf in inverted_index[t]:
                    candidate_scores[pool_id] += idf

        if not candidate_scores:
            continue

        sorted_cands = sorted(candidate_scores.items(), key=lambda x: -x[1])[:top_k]
        for rank, (pool_id, score) in enumerate(sorted_cands):
            results.append((s1.entity_id, pool_id, rank, score))

    return results
