"""Channel C2: Scalable character TF-IDF top-K retrieval (Section 12.2)."""
import numpy as np
from typing import Dict, List, Sequence, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from ber.io.schemas import CanonicalRecord


def retrieve_c2_tfidf(
    s1_records: Sequence[CanonicalRecord],
    pool_records: Sequence[CanonicalRecord],
    top_k_per_source: int = 50,
    chunk_size: int = 1000,
) -> List[Tuple[str, str, int, float]]:
    """C2: Sparse top-K character n-gram TF-IDF cosine retrieval.

    Evaluated separately per source (S2 and S3) to guarantee balanced retrieval budget.
    Uses sublinear TF, char_wb (2,4) n-grams, and row-chunked matrix multiplication.
    """
    if len(s1_records) == 0 or len(pool_records) == 0:
        return []

    # Partition pool records by source (S2 and S3)
    s2_records = [r for r in pool_records if r.source == "S2"]
    s3_records = [r for r in pool_records if r.source == "S3"]

    results: List[Tuple[str, str, int, float]] = []

    for pool_subset in [s2_records, s3_records]:
        if not pool_subset:
            continue

        all_names = [r.name_norm or "empty" for r in s1_records] + [
            r.name_norm or "empty" for r in pool_subset
        ]

        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            sublinear_tf=True,
            min_df=1,
        )
        vec.fit(all_names)

        s1_mat = vec.transform([r.name_norm or "empty" for r in s1_records])
        pool_mat = vec.transform([r.name_norm or "empty" for r in pool_subset])

        n_s1 = s1_mat.shape[0]
        n_pool = pool_mat.shape[0]
        actual_k = min(top_k_per_source, n_pool)

        pool_mat_t = pool_mat.T.tocsr()

        for chunk_start in range(0, n_s1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_s1)
            chunk_sims = (s1_mat[chunk_start:chunk_end] * pool_mat_t).toarray()

            for i_local in range(chunk_sims.shape[0]):
                s1_idx = chunk_start + i_local
                s1_id = s1_records[s1_idx].entity_id
                row_sims = chunk_sims[i_local]

                if actual_k == n_pool:
                    top_indices = np.argsort(-row_sims)
                else:
                    top_part = np.argpartition(-row_sims, actual_k)[:actual_k]
                    top_indices = top_part[np.argsort(-row_sims[top_part])]

                for rank, pool_idx in enumerate(top_indices):
                    sim = float(row_sims[pool_idx])
                    if sim > 0.05:  # Filter out completely disjoint noise
                        pid = pool_subset[pool_idx].entity_id
                        results.append((s1_id, pid, rank, sim))

    return results
