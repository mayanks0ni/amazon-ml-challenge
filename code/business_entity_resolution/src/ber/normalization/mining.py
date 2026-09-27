"""Abbreviation mining and alignment (Section 10.4)."""
from collections import defaultdict
import re
from typing import Dict, List, Sequence, Set, Tuple


def get_consonant_skeleton(word: str) -> str:
    """Extracts consonant skeleton of a word (e.g. 'corporation' -> 'crprtn')."""
    return "".join(c for c in word.lower() if c in "bcdfghjklmnpqrstvwxyz")


def is_abbrev(a: str, b: str) -> bool:
    """Tests if string a could be an abbreviation of string b (or vice-versa).

    Criteria from Section 10.4:
    - a is a prefix of b
    - or a is a subsequence of b's consonant skeleton
    - or first letters match and len(a) <= 4 and all letters of a appear in b in order.
    """
    if not a or not b or a == b:
        return False

    short, long_w = (a, b) if len(a) <= len(b) else (b, a)

    # 1. Prefix check
    if long_w.startswith(short) and len(short) >= 2:
        return True

    # 2. Consonant skeleton subsequence
    skel = get_consonant_skeleton(long_w)
    if short in skel and len(short) >= 2:
        return True

    # 3. Subsequence with first letter match
    if short[0] == long_w[0] and len(short) <= 4:
        it = iter(long_w)
        if all(c in it for c in short):
            return True

    return False


def compute_abbrev_align(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    """Computes abbrev_align feature: fraction of unaligned tokens on shorter side
    that are explained by prefix or consonant-skeleton of an unaligned token on the other side.

    Essential for France / unseen countries where training abbreviations were not mined.
    """
    set_a = set(tokens_a)
    set_b = set(tokens_b)

    unaligned_a = [t for t in set_a - set_b if len(t) >= 2]
    unaligned_b = [t for t in set_b - set_a if len(t) >= 2]

    if not unaligned_a or not unaligned_b:
        return 0.0

    shorter_list, longer_list = (
        (unaligned_a, unaligned_b)
        if len(unaligned_a) <= len(unaligned_b)
        else (unaligned_b, unaligned_a)
    )

    matched = 0
    for s_tok in shorter_list:
        for l_tok in longer_list:
            if is_abbrev(s_tok, l_tok):
                matched += 1
                break

    return float(matched / len(shorter_list))


def mine_abbreviations_from_pairs(
    matched_pairs: Sequence[Tuple[str, str]],
    blocked_negatives: Sequence[Tuple[str, str]],
    min_count: int = 5,
    min_precision: float = 0.90,
) -> Dict[str, str]:
    """Mines high-precision abbreviation pairs strictly from training ground truth (Section 10.4)."""
    co_occur_pos: Dict[Tuple[str, str], int] = defaultdict(int)
    co_occur_neg: Dict[Tuple[str, str], int] = defaultdict(int)

    for name_a, name_b in matched_pairs:
        toks_a = set(name_a.split())
        toks_b = set(name_b.split())
        unaligned_a = toks_a - toks_b
        unaligned_b = toks_b - toks_a
        for a in unaligned_a:
            for b in unaligned_b:
                if is_abbrev(a, b):
                    key = (min(a, b), max(a, b))
                    co_occur_pos[key] += 1

    for name_a, name_b in blocked_negatives:
        toks_a = set(name_a.split())
        toks_b = set(name_b.split())
        unaligned_a = toks_a - toks_b
        unaligned_b = toks_b - toks_a
        for a in unaligned_a:
            for b in unaligned_b:
                key = (min(a, b), max(a, b))
                if key in co_occur_pos:
                    co_occur_neg[key] += 1

    learned: Dict[str, str] = {}
    for (a, b), pos_cnt in co_occur_pos.items():
        neg_cnt = co_occur_neg.get((a, b), 0)
        precision = pos_cnt / (pos_cnt + neg_cnt)
        if pos_cnt >= min_count and precision >= min_precision:
            short, long_w = (a, b) if len(a) < len(b) else (b, a)
            learned[short] = long_w

    return learned
