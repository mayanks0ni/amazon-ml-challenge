"""Ground truth validation and decision gates evaluation (Section 4.4)."""
from collections import Counter, defaultdict
import re
from typing import Dict, List, Set, Tuple
from ber.io.schemas import RawRecord


def evaluate_decision_gates(
    s1_records: List[RawRecord],
    s2_records: List[RawRecord],
    s3_records: List[RawRecord],
    ground_truth: Dict[str, List[str]],
) -> Dict[str, any]:
    """Computes EDA statistics and evaluates decision gates G1 through G6.

    Gate G1: No S2/S3 ID appears in > 1 GT row -> Hard exclusivity
    Gate G2: >= 99.5% GT pairs have equal normalized country -> Country partitioned blocking
    Gate G3: Postal-like token present in >= 60% of records -> Numeric channel is primary
    Gate G4: Embedding channel adds >= 0.3 pp pair recall (placeholder/evaluated during blocking)
    Gate G5: S2 vs S3 feature distribution difference -> Source indicator feature
    Gate G6: Share of multi-match S1s >= 15% -> Collective stage 2b high priority
    """
    id_to_record = {r.entity_id: r for r in (s1_records + s2_records + s3_records)}

    # Gate G1: Exclusivity count
    pool_id_counts: Counter = Counter()
    total_links = 0
    multi_match_s1_count = 0
    singleton_count = 0

    for s1_id, matches in ground_truth.items():
        if len(matches) == 0:
            singleton_count += 1
        elif len(matches) > 1:
            multi_match_s1_count += 1

        for m in matches:
            pool_id_counts[m] += 1
            total_links += 1

    shared_pool_records = {k: v for k, v in pool_id_counts.items() if v > 1}
    g1_hard_exclusivity = len(shared_pool_records) == 0

    # Gate G2: Same country rate on GT pairs
    same_country_pairs = 0
    checked_pairs = 0
    for s1_id, matches in ground_truth.items():
        s1_rec = id_to_record.get(s1_id)
        if not s1_rec:
            continue
        s1_c = s1_rec.country.strip().lower()
        for m in matches:
            m_rec = id_to_record.get(m)
            if not m_rec:
                continue
            m_c = m_rec.country.strip().lower()
            checked_pairs += 1
            if s1_c == m_c:
                same_country_pairs += 1

    g2_country_rate = (same_country_pairs / checked_pairs) if checked_pairs > 0 else 1.0
    g2_partition_by_country = g2_country_rate >= 0.995

    # Gate G3: Postal code presence rate
    postal_re = re.compile(r"\b\d{5,6}\b|\b\d{5}-\d{4}\b")
    all_records = s1_records + s2_records + s3_records
    records_with_postal = sum(
        1 for r in all_records if postal_re.search(r.business_address)
    )
    g3_postal_rate = (records_with_postal / len(all_records)) if all_records else 0.0
    g3_numeric_primary = g3_postal_rate >= 0.60

    # Gate G6: Share of multi-match S1s
    total_s1 = len(ground_truth)
    g6_multi_match_share = (multi_match_s1_count / total_s1) if total_s1 > 0 else 0.0
    g6_stage2b_priority = g6_multi_match_share >= 0.15

    return {
        "G1_hard_exclusivity": g1_hard_exclusivity,
        "G1_conflicting_pool_records": len(shared_pool_records),
        "G2_country_rate": g2_country_rate,
        "G2_partition_by_country": g2_partition_by_country,
        "G3_postal_rate": g3_postal_rate,
        "G3_numeric_primary": g3_numeric_primary,
        "G6_multi_match_share": g6_multi_match_share,
        "G6_stage2b_priority": g6_stage2b_priority,
        "total_s1": total_s1,
        "total_singletons": singleton_count,
        "singleton_rate": (singleton_count / total_s1) if total_s1 > 0 else 0.0,
        "total_links": total_links,
    }
