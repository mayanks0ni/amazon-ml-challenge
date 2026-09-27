"""Channel C5: Address and numeric key blocking."""
from collections import Counter, defaultdict
import math
from typing import Dict, List, Sequence, Set, Tuple
from ber.io.schemas import CanonicalRecord

ADDR_STOP_WORDS = {
    "delhi", "mumbai", "india", "road", "street", "st", "rd", "ave", "avenue",
    "floor", "flr", "nagar", "near", "opposite", "opp", "bldg", "building",
    "flat", "plot", "shop", "no", "andhra", "pradesh", "maharashtra", "karnataka",
    "tamil", "nadu", "state", "dist", "district", "city", "lane", "ln", "court",
    "ct", "block", "sector", "sec", "phase", "cross", "main", "layout", "colony",
    "marg", "gali", "galli", "hwy", "highway", "dr", "drive", "house", "null",
    "township", "c", "o", "co", "pvt", "ltd", "limited", "private"
}


def retrieve_c5_address_numeric(
    s1_records: Sequence[CanonicalRecord],
    pool_records: Sequence[CanonicalRecord],
    top_k: int = 30,
    max_df: int = 250,
) -> List[Tuple[str, str, int, float]]:
    """C5: Address exact keys, shared numeric identifiers, and address token IDF inverted index.
    Catches businesses with different/rebranded trade names at the identical physical location.
    """
    index_postal_house: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    index_postal_name3: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    index_clean_addr: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    index_sorted_addr: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    total_docs = len(s1_records) + len(pool_records)
    df_counter: Counter = Counter()

    for rec in pool_records:
        if rec.postal:
            if rec.house_no:
                index_postal_house[(rec.postal, rec.house_no)].append(rec.entity_id)
            if len(rec.name_core) >= 3:
                index_postal_name3[(rec.postal, rec.name_core[:3])].append(rec.entity_id)
        if rec.addr_clean and len(rec.addr_clean) >= 10 and rec.country_norm:
            cplist = index_clean_addr[(rec.addr_clean[:25], rec.country_norm)]
            if len(cplist) < 15:
                cplist.append(rec.entity_id)
        if rec.sorted_addr and len(rec.sorted_addr) >= 6 and rec.country_norm:
            splist = index_sorted_addr[(rec.sorted_addr, rec.country_norm)]
            if len(splist) < 15:
                splist.append(rec.entity_id)

        # Count tokens for address inverted index
        tokens = set(t for t in rec.addr_tokens if t not in ADDR_STOP_WORDS and len(t) >= 2)
        tokens.update(rec.addr_nums)
        for t in tokens:
            df_counter[t] += 1

    # Inverted index on rare address tokens and numbers
    addr_inverted_index: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for rec in pool_records:
        tokens = set(t for t in rec.addr_tokens if t not in ADDR_STOP_WORDS and len(t) >= 2)
        tokens.update(rec.addr_nums)
        for t in tokens:
            if df_counter[t] <= max_df:
                idf = math.log((total_docs + 1.0) / (df_counter[t] + 1.0)) + 1.0
                addr_inverted_index[t].append((rec.entity_id, idf))

    results: List[Tuple[str, str, int, float]] = []

    for s1 in s1_records:
        candidate_scores: Dict[str, float] = {}

        # 1. Exact postal + house
        if s1.postal and s1.house_no:
            for pid in index_postal_house.get((s1.postal, s1.house_no), []):
                candidate_scores[pid] = max(candidate_scores.get(pid, 0.0), 10.0)

        # 2. Exact postal + name prefix 3
        if s1.postal and len(s1.name_core) >= 3:
            for pid in index_postal_name3.get((s1.postal, s1.name_core[:3]), []):
                candidate_scores[pid] = max(candidate_scores.get(pid, 0.0), 8.0)

        # 3. Clean address prefix 25
        if s1.addr_clean and len(s1.addr_clean) >= 10 and s1.country_norm:
            for pid in index_clean_addr.get((s1.addr_clean[:25], s1.country_norm), []):
                candidate_scores[pid] = max(candidate_scores.get(pid, 0.0), 7.0)

        # 4. Sorted address
        if s1.sorted_addr and len(s1.sorted_addr) >= 6 and s1.country_norm:
            for pid in index_sorted_addr.get((s1.sorted_addr, s1.country_norm), []):
                candidate_scores[pid] = max(candidate_scores.get(pid, 0.0), 6.0)

        # 5. Address token IDF inverted index
        s1_tokens = set(t for t in s1.addr_tokens if t not in ADDR_STOP_WORDS and len(t) >= 2)
        s1_tokens.update(s1.addr_nums)
        if s1_tokens:
            cand_token_scores: Dict[str, float] = defaultdict(float)
            cand_token_counts: Dict[str, int] = defaultdict(int)
            for t in s1_tokens:
                if t in addr_inverted_index:
                    for pid, idf in addr_inverted_index[t]:
                        cand_token_scores[pid] += idf
                        cand_token_counts[pid] += 1

            for pid, score in cand_token_scores.items():
                # Require >= 2 shared tokens OR 1 token + shared address number
                if cand_token_counts[pid] >= 2 or (cand_token_counts[pid] >= 1 and any(n in s1.addr_nums for n in s1_tokens)):
                    candidate_scores[pid] = max(candidate_scores.get(pid, 0.0), score)

        if not candidate_scores:
            continue

        sorted_cands = sorted(candidate_scores.items(), key=lambda x: -x[1])[:top_k]
        for rank, (pid, score) in enumerate(sorted_cands):
            results.append((s1.entity_id, pid, rank, score))

    return results
