"""Token rarity (IDF) statistics computation fit per country (Section 13.5)."""
from collections import Counter
import math
from typing import Dict, List, Sequence, Set
from ber.io.schemas import CanonicalRecord


class CorpusIDF:
    """Computes and stores token IDFs for a given corpus (fit per split/country)."""

    def __init__(self, records: Sequence[CanonicalRecord]):
        self.total_docs = len(records)
        self.doc_freq: Counter = Counter()

        for rec in records:
            # Count unique tokens in name and address
            tokens = set(rec.name_core_tokens).union(set(rec.addr_tokens))
            for t in tokens:
                self.doc_freq[t] += 1

        self.default_idf = math.log(self.total_docs + 1.0) + 1.0

    def get_idf(self, token: str) -> float:
        """Returns IDF of token with add-1 smoothing."""
        df = self.doc_freq.get(token, 0)
        if df == 0:
            return self.default_idf
        return math.log((self.total_docs + 1.0) / (df + 1.0)) + 1.0


def compute_idf_shared_unshared(
    tokens_a: Sequence[str],
    tokens_b: Sequence[str],
    idf_model: CorpusIDF,
) -> Dict[str, float]:
    """Computes shared and unshared IDF metrics.
    Shared rare tokens are strong positive evidence;
    Unshared rare tokens are strong negative evidence (e.g. branch names).
    """
    set_a = set(tokens_a)
    set_b = set(tokens_b)

    shared = set_a.intersection(set_b)
    unshared_a = set_a - set_b
    unshared_b = set_b - set_a

    shared_idfs = [idf_model.get_idf(t) for t in shared]
    unshared_a_idfs = [idf_model.get_idf(t) for t in unshared_a]
    unshared_b_idfs = [idf_model.get_idf(t) for t in unshared_b]

    shared_sum = sum(shared_idfs)
    shared_max = max(shared_idfs) if shared_idfs else 0.0

    unshared_a_sum = sum(unshared_a_idfs)
    unshared_b_sum = sum(unshared_b_idfs)
    unshared_max = max(unshared_a_idfs + unshared_b_idfs) if (unshared_a_idfs or unshared_b_idfs) else 0.0

    return {
        "idf_shared_sum": shared_sum,
        "idf_shared_max": shared_max,
        "idf_unshared_sum_a": unshared_a_sum,
        "idf_unshared_sum_b": unshared_b_sum,
        "idf_unshared_max": unshared_max,
    }
