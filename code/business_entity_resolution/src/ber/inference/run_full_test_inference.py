"""Full test dataset inference and submission generation.

Scalable CPFC pipeline that:
1. Samples real training data, canonicalizes, blocks, extracts features, trains model.
2. Indexes test pool with inverted indices for efficient blocking.
3. For each test S1: retrieves candidates → extracts features → scores with model.
4. Applies exclusivity + expected-F0.5 set selection.
5. Writes matching_results.tsv and candidate_pairs.tsv.
"""
import csv
import gc
import math
import os
import pickle
import random
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz

from ber.io.schemas import RawRecord, CanonicalRecord, CandidateProvenance
from ber.normalization.pipeline import canonicalize_record
from ber.normalization.unicode import normalize_unicode
from ber.normalization.indic import transliterate_indic
from ber.normalization.fold import fold_text
from ber.extraction.legal_form import extract_legal_form, check_legal_form_compatibility
from ber.extraction.postal import extract_postal_code, compare_postal_codes
from ber.extraction.numbers import extract_address_numbers, compare_house_numbers, compare_unit_numbers
from ber.normalization.mining import compute_abbrev_align
from ber.decision.exclusivity import apply_exclusivity
from ber.decision.expected_f import select_expected_f05


# ────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────
ADDR_STOP_WORDS = {
    "st", "street", "ave", "avenue", "rd", "road", "blvd", "dr", "drive",
    "ln", "lane", "ct", "court", "pl", "place", "terrace", "null", "township",
    "apt", "suite", "ste", "flr", "floor", "delhi", "mumbai", "india", "nagar",
    "near", "opposite", "opp", "bldg", "building", "flat", "plot", "shop", "no",
    "andhra", "pradesh", "maharashtra", "karnataka", "tamil", "nadu", "state",
    "dist", "district", "city", "block", "sector", "sec", "phase", "cross", "main",
    "layout", "colony", "marg", "gali", "galli", "hwy", "highway", "house",
    "c", "o", "co", "pvt", "ltd", "limited", "private", "rue", "boulevard",
    "paris", "france", "cedex", "arrondissement",
}

TRAIN_DIR = "student_resource/dataset/train"
TEST_DIR = "student_resource/dataset/test"
OUTPUT_DIR = "output"
MODEL_DIR = "output/models"
SAMPLE_SIZE = 50_000  # S1 entities to sample for training
POOL_SAMPLE_SIZE = 300_000  # Max pool records to load for training (beyond GT positives)
BUDGET_M = 100


# ────────────────────────────────────────────────────────────────
# Lightweight record representation for scalability
# ────────────────────────────────────────────────────────────────
class LightRecord:
    """Lightweight record for efficient indexing and feature extraction."""
    __slots__ = [
        "entity_id", "name_raw", "addr_raw", "country",
        "name_norm", "name_core", "name_core_tokens", "name_fold",
        "name_clean", "legal_form", "name_acronym",
        "addr_norm", "addr_tokens", "addr_fold", "addr_nums",
        "addr_clean", "sorted_addr", "postal", "house_no", "unit",
        "landmark", "addr_tail", "source",
    ]

    def __init__(self, entity_id, name_raw, addr_raw, country, source=""):
        self.entity_id = entity_id
        self.name_raw = name_raw
        self.addr_raw = addr_raw
        self.country = country.strip()
        self.source = source

        # Name normalization
        name = transliterate_indic(name_raw)
        norm_name = normalize_unicode(name)
        norm_name = re.sub(r"https?://", "", norm_name)
        norm_name = re.sub(r"\bwww\.", "", norm_name)
        norm_name = re.sub(
            r"\.(?:com|org|net|co\.in|in|co|io|biz|info|gov|edu|fr|us)(?=[/\s]|$)",
            "", norm_name
        ).strip()

        core, lf = extract_legal_form(norm_name)
        self.name_norm = norm_name
        self.name_core = core or ""
        self.legal_form = lf or ""
        self.name_fold = fold_text(self.name_core) if self.name_core else ""
        self.name_clean = "".join(ch for ch in self.name_core if ch.isalnum())
        self.name_core_tokens = [t for t in self.name_core.split() if len(t) >= 2]
        self.name_acronym = "".join(t[0] for t in self.name_core_tokens).lower() if len(self.name_core_tokens) >= 2 else ""

        # Address normalization
        addr = transliterate_indic(addr_raw)
        self.addr_norm = addr.strip()
        self.addr_fold = fold_text(self.addr_norm) if self.addr_norm else ""
        self.postal = extract_postal_code(addr, country=self.country)
        nums, house, unit = extract_address_numbers(addr)
        self.addr_nums = nums
        self.house_no = house or ""
        self.unit = unit or ""
        self.addr_clean = "".join(ch for ch in addr.lower() if ch.isalnum())

        # Sorted address key
        s_ord = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", addr.lower())
        s_toks = re.findall(r"[a-z0-9]+", s_ord)
        meaningful = sorted([t for t in s_toks if t not in ADDR_STOP_WORDS])
        self.sorted_addr = "".join(meaningful[:5])
        self.addr_tokens = [t for t in s_toks if len(t) >= 2 and t not in ADDR_STOP_WORDS]
        self.landmark = ""
        self.addr_tail = " ".join(s_toks[-3:]) if len(s_toks) >= 3 else " ".join(s_toks)


# ────────────────────────────────────────────────────────────────
# Lightweight IDF
# ────────────────────────────────────────────────────────────────
class LightIDF:
    """Lightweight IDF computation using dict counters."""

    def __init__(self):
        self.doc_freq = Counter()
        self.total_docs = 0
        self.default_idf = 1.0

    def add_record(self, rec: LightRecord):
        tokens = set(rec.name_core_tokens) | set(rec.addr_tokens)
        for t in tokens:
            self.doc_freq[t] += 1
        self.total_docs += 1

    def finalize(self):
        self.default_idf = math.log(self.total_docs + 1.0) + 1.0

    def get_idf(self, token: str) -> float:
        df = self.doc_freq.get(token, 0)
        if df == 0:
            return self.default_idf
        return math.log((self.total_docs + 1.0) / (df + 1.0)) + 1.0


# ────────────────────────────────────────────────────────────────
# Feature extraction for a pair of LightRecords
# ────────────────────────────────────────────────────────────────
def extract_pair_features(
    rec_a: LightRecord,
    rec_b: LightRecord,
    idf: LightIDF,
    n_channels: int = 0,
    cand_count: int = 1,
) -> Dict[str, float]:
    """Extracts similarity features for a (S1, pool) candidate pair."""
    feats = {}

    # ── Name features ──
    feats["name_exact_norm"] = 1.0 if rec_a.name_norm and rec_a.name_norm == rec_b.name_norm else 0.0
    feats["name_exact_core"] = 1.0 if rec_a.name_core and rec_a.name_core == rec_b.name_core else 0.0
    feats["name_exact_fold"] = 1.0 if rec_a.name_fold and rec_a.name_fold == rec_b.name_fold else 0.0

    feats["name_lev_ratio_norm"] = distance.Levenshtein.normalized_similarity(rec_a.name_norm, rec_b.name_norm)
    feats["name_lev_ratio_core"] = distance.Levenshtein.normalized_similarity(rec_a.name_core, rec_b.name_core)
    feats["name_jw_core"] = distance.JaroWinkler.similarity(rec_a.name_core, rec_b.name_core)
    feats["name_jw_fold"] = distance.JaroWinkler.similarity(rec_a.name_fold, rec_b.name_fold)

    feats["name_token_set_ratio_core"] = fuzz.token_set_ratio(rec_a.name_core, rec_b.name_core) / 100.0
    feats["name_token_sort_ratio_core"] = fuzz.token_sort_ratio(rec_a.name_core, rec_b.name_core) / 100.0
    feats["name_partial_ratio_core"] = fuzz.partial_ratio(rec_a.name_core, rec_b.name_core) / 100.0

    # Token Jaccard
    set_a = set(rec_a.name_core_tokens)
    set_b = set(rec_b.name_core_tokens)
    union_len = len(set_a | set_b)
    feats["name_token_jaccard_core"] = (len(set_a & set_b) / union_len) if union_len > 0 else 0.0

    # Soft TF-IDF (simplified)
    soft_score = 0.0
    for ta in rec_a.name_core_tokens:
        best_sim = 0.0
        best_tb = ""
        for tb in rec_b.name_core_tokens:
            sim = distance.JaroWinkler.similarity(ta, tb)
            if sim > best_sim:
                best_sim = sim
                best_tb = tb
        if best_sim >= 0.90:
            soft_score += idf.get_idf(ta) * idf.get_idf(best_tb) * best_sim
    norm_a = sum(idf.get_idf(t) ** 2 for t in rec_a.name_core_tokens) ** 0.5
    norm_b = sum(idf.get_idf(t) ** 2 for t in rec_b.name_core_tokens) ** 0.5
    feats["name_soft_tfidf"] = (soft_score / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0

    # IDF shared/unshared
    shared = set_a & set_b
    unshared_a = set_a - set_b
    unshared_b = set_b - set_a
    shared_idfs = [idf.get_idf(t) for t in shared]
    unshared_a_idfs = [idf.get_idf(t) for t in unshared_a]
    unshared_b_idfs = [idf.get_idf(t) for t in unshared_b]
    feats["idf_shared_sum"] = sum(shared_idfs)
    feats["idf_shared_max"] = max(shared_idfs) if shared_idfs else 0.0
    feats["idf_unshared_max"] = max(unshared_a_idfs + unshared_b_idfs) if (unshared_a_idfs or unshared_b_idfs) else 0.0

    # First/last token
    feats["name_first_token_eq"] = 1.0 if rec_a.name_core_tokens and rec_b.name_core_tokens and rec_a.name_core_tokens[0] == rec_b.name_core_tokens[0] else 0.0
    feats["name_last_token_eq"] = 1.0 if rec_a.name_core_tokens and rec_b.name_core_tokens and rec_a.name_core_tokens[-1] == rec_b.name_core_tokens[-1] else 0.0

    # Acronym
    acronym_match = 0.0
    if rec_a.name_acronym and rec_b.name_core_tokens:
        if rec_a.name_acronym in rec_b.name_core_tokens or rec_a.name_acronym == rec_b.name_acronym:
            acronym_match = 1.0
    if rec_b.name_acronym and rec_a.name_core_tokens:
        if rec_b.name_acronym in rec_a.name_core_tokens or rec_b.name_acronym == rec_a.name_acronym:
            acronym_match = 1.0
    feats["acronym_match"] = acronym_match

    # Legal form
    lf_eq, lf_conflict = check_legal_form_compatibility(rec_a.legal_form, rec_b.legal_form)
    feats["legal_form_eq"] = float(lf_eq)
    feats["legal_form_conflict"] = float(lf_conflict)

    # Name length ratios
    len_a = len(rec_a.name_core)
    len_b = len(rec_b.name_core)
    max_len = max(len_a, len_b)
    feats["name_len_ratio_chars"] = (min(len_a, len_b) / max_len) if max_len > 0 else 1.0
    tok_a = len(rec_a.name_core_tokens)
    tok_b = len(rec_b.name_core_tokens)
    max_tok = max(tok_a, tok_b)
    feats["name_len_ratio_tokens"] = (min(tok_a, tok_b) / max_tok) if max_tok > 0 else 1.0

    # Digits in name
    digits_a = set(c for c in rec_a.name_raw if c.isdigit())
    digits_b = set(c for c in rec_b.name_raw if c.isdigit())
    if digits_a and digits_b:
        feats["digits_in_name_conflict"] = 1.0 if digits_a != digits_b else 0.0
    else:
        feats["digits_in_name_conflict"] = 0.0

    # Alt name max sim (using core only for speed)
    feats["alt_name_max_sim"] = feats["name_token_set_ratio_core"]

    # ── Address features ──
    feats["addr_exact_eq"] = 1.0 if rec_a.addr_norm and rec_a.addr_norm == rec_b.addr_norm else 0.0
    feats["addr_lev_ratio"] = distance.Levenshtein.normalized_similarity(rec_a.addr_norm, rec_b.addr_norm)
    feats["addr_token_set_ratio"] = fuzz.token_set_ratio(rec_a.addr_norm, rec_b.addr_norm) / 100.0
    feats["addr_token_sort_ratio"] = fuzz.token_sort_ratio(rec_a.addr_norm, rec_b.addr_norm) / 100.0
    feats["addr_fold_sim"] = fuzz.token_set_ratio(rec_a.addr_fold, rec_b.addr_fold) / 100.0

    # Postal
    p_eq, p_both, p_pref = compare_postal_codes(rec_a.postal, rec_b.postal)
    feats["postal_eq"] = float(p_eq)
    feats["postal_both_present"] = float(p_both)
    feats["postal_conflict"] = 1.0 if p_both == 1 and p_eq == 0 else 0.0

    # House number
    h_eq, h_conflict = compare_house_numbers(rec_a.house_no, rec_b.house_no)
    feats["house_eq"] = float(h_eq)
    feats["house_conflict"] = float(h_conflict)

    # Unit
    u_eq, u_conflict = compare_unit_numbers(rec_a.unit, rec_b.unit)
    feats["unit_eq"] = float(u_eq)
    feats["unit_conflict"] = float(u_conflict)

    # Numbers jaccard
    set_nums_a = set(rec_a.addr_nums)
    set_nums_b = set(rec_b.addr_nums)
    if set_nums_a and set_nums_b:
        inter = len(set_nums_a & set_nums_b)
        union = len(set_nums_a | set_nums_b)
        feats["nums_jaccard"] = (inter / union) if union > 0 else 0.0
    else:
        feats["nums_jaccard"] = 0.0

    # Address IDF shared
    addr_set_a = set(rec_a.addr_tokens)
    addr_set_b = set(rec_b.addr_tokens)
    addr_shared = addr_set_a & addr_set_b
    addr_unshared = (addr_set_a - addr_set_b) | (addr_set_b - addr_set_a)
    addr_shared_idfs = [idf.get_idf(t) for t in addr_shared]
    addr_unshared_idfs = [idf.get_idf(t) for t in addr_unshared]
    feats["addr_idf_shared_max"] = max(addr_shared_idfs) if addr_shared_idfs else 0.0
    feats["addr_idf_unshared_sum"] = sum(addr_unshared_idfs)

    # Address length ratio
    addr_len_a = len(rec_a.addr_norm)
    addr_len_b = len(rec_b.addr_norm)
    max_al = max(addr_len_a, addr_len_b)
    feats["addr_len_ratio"] = (min(addr_len_a, addr_len_b) / max_al) if max_al > 0 else 1.0

    # ── Cross features ──
    feats["same_country"] = 1.0 if rec_a.country and rec_b.country and rec_a.country == rec_b.country else 0.0
    feats["is_s3"] = 1.0 if rec_b.source == "S3" else 0.0
    feats["n_channels"] = float(n_channels)
    feats["cand_count_i"] = float(cand_count)

    name_sim = feats["name_token_set_ratio_core"]
    addr_sim = feats["addr_token_set_ratio"]
    feats["name_sim_x_addr_sim"] = name_sim * addr_sim
    feats["name_hi_addr_lo"] = 1.0 if (name_sim >= 0.85 and addr_sim < 0.40) else 0.0
    feats["name_lo_addr_hi"] = 1.0 if (name_sim < 0.40 and addr_sim >= 0.85) else 0.0

    return feats


# ────────────────────────────────────────────────────────────────
# Blocking via inverted indices
# ────────────────────────────────────────────────────────────────
def build_pool_index(pool: Dict[str, LightRecord]) -> Dict[str, dict]:
    """Builds inverted indices over pool records for efficient blocking."""
    idx = {
        "core_country": defaultdict(list),
        "fold_country": defaultdict(list),
        "clean_country": defaultdict(list),
        "token": defaultdict(list),
        "postal_house": defaultdict(list),
        "clean_addr": defaultdict(list),
        "sorted_addr": defaultdict(list),
    }

    for pid, rec in pool.items():
        c = rec.country
        core = rec.name_core
        if core:
            postings = idx["core_country"][(core, c)]
            if len(postings) < 50:
                postings.append(pid)

            fold = rec.name_fold
            if fold and fold != core:
                fp = idx["fold_country"][(fold, c)]
                if len(fp) < 50:
                    fp.append(pid)

            clean = rec.name_clean
            if clean:
                cp = idx["clean_country"][(clean, c)]
                if len(cp) < 50:
                    cp.append(pid)

            for t in rec.name_core_tokens:
                tp = idx["token"][(t, c)]
                if len(tp) < 400:
                    tp.append(pid)

        postal = rec.postal
        house = rec.house_no
        if postal and house:
            lp = idx["postal_house"][(postal, house, c)]
            if len(lp) < 20:
                lp.append(pid)

        ca = rec.addr_clean
        if len(ca) >= 10:
            cap = idx["clean_addr"][(ca[:25], c)]
            if len(cap) < 20:
                cap.append(pid)

        sa = rec.sorted_addr
        if len(sa) >= 6:
            sap = idx["sorted_addr"][(sa, c)]
            if len(sap) < 20:
                sap.append(pid)

    return idx


def retrieve_candidates(rec: LightRecord, idx: dict) -> List[str]:
    """Retrieves candidate pool IDs for an S1 record using inverted indices."""
    c = rec.country
    core = rec.name_core
    cand_set: Dict[str, int] = defaultdict(int)  # pid -> channel count

    if core:
        for pid in idx["core_country"].get((core, c), []):
            cand_set[pid] += 1
        fold = rec.name_fold
        if fold and fold != core:
            for pid in idx["fold_country"].get((fold, c), []):
                cand_set[pid] += 1
        clean = rec.name_clean
        if clean:
            for pid in idx["clean_country"].get((clean, c), []):
                cand_set[pid] += 1
        core_tokens = rec.name_core_tokens
        if core_tokens:
            req_tokens = 1 if len(core_tokens) == 1 else 2
            tok_counts: Dict[str, int] = defaultdict(int)
            for t in core_tokens:
                for pid in idx["token"].get((t, c), []):
                    tok_counts[pid] += 1
            for pid, cnt in tok_counts.items():
                if cnt >= req_tokens:
                    cand_set[pid] += 1

    postal = rec.postal
    house = rec.house_no
    if postal and house:
        for pid in idx["postal_house"].get((postal, house, c), []):
            cand_set[pid] += 1

    ca = rec.addr_clean
    if len(ca) >= 10:
        for pid in idx["clean_addr"].get((ca[:25], c), []):
            cand_set[pid] += 1

    sa = rec.sorted_addr
    if len(sa) >= 6:
        for pid in idx["sorted_addr"].get((sa, c), []):
            cand_set[pid] += 1

    # Sort by channel count (more channels = higher priority), cap at budget
    scored = sorted(cand_set.items(), key=lambda x: -x[1])
    return [pid for pid, _ in scored[:BUDGET_M]]


# ────────────────────────────────────────────────────────────────
# Training on real data
# ────────────────────────────────────────────────────────────────
def load_records_streaming(path: str, source: str) -> List[LightRecord]:
    """Loads records from TSV file into LightRecord objects."""
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                records.append(LightRecord(
                    entity_id=parts[0],
                    name_raw=parts[1],
                    addr_raw=parts[2],
                    country=parts[3],
                    source=source,
                ))
    return records


def load_ground_truth_streaming(path: str) -> Dict[str, List[str]]:
    """Loads ground truth from TSV."""
    gt = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 2:
                sid = parts[0]
                matches = [m.strip() for m in parts[1].split(",") if m.strip()] if parts[1].strip() else []
                gt[sid] = matches
            elif len(parts) == 1:
                gt[parts[0]] = []
    return gt


def train_model(
    train_dir: str,
    sample_size: int = SAMPLE_SIZE,
    seed: int = 42,
) -> Tuple[List[lgb.Booster], List[str], "LightIDF", dict]:
    """Trains LightGBM model on sampled real training data.

    Returns: (models, feature_cols, idf_model, optimal_params)
    """
    random.seed(seed)
    np.random.seed(seed)

    print(f"\n[Train] Loading training data from {train_dir}...")
    t0 = time.time()

    # Load ground truth
    gt = load_ground_truth_streaming(os.path.join(train_dir, "train_ground_truth.tsv"))
    print(f"  Ground truth loaded: {len(gt):,} S1 entities ({time.time()-t0:.1f}s)")

    # Sample S1 entities
    all_s1_ids = list(gt.keys())
    if len(all_s1_ids) > sample_size:
        sampled_s1_ids = set(random.sample(all_s1_ids, sample_size))
    else:
        sampled_s1_ids = set(all_s1_ids)
    print(f"  Sampled {len(sampled_s1_ids):,} S1 entities for training")

    # Collect needed pool IDs from ground truth
    needed_pool_ids: Set[str] = set()
    for sid in sampled_s1_ids:
        needed_pool_ids.update(gt.get(sid, []))

    # Load S1 records (sampled only)
    print(f"  Loading S1 records...")
    t1 = time.time()
    s1_all = {}
    with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and parts[0] in sampled_s1_ids:
                rec = LightRecord(parts[0], parts[1], parts[2], parts[3], source="S1")
                s1_all[parts[0]] = rec
    print(f"    Loaded {len(s1_all):,} S1 records ({time.time()-t1:.1f}s)")

    # Load pool records (S2 + S3)
    # Two-pass approach: first collect GT-positive line numbers + reservoir sample, then load only those
    print(f"  Loading pool records (GT positives + {POOL_SAMPLE_SIZE:,} random sample)...")
    t1 = time.time()
    pool_all: Dict[str, LightRecord] = {}

    for src_num, src_name in [(2, "S2"), (3, "S3")]:
        path = os.path.join(train_dir, f"train_source{src_num}.tsv")

        # Pass 1: Find line numbers for GT-positive IDs + reservoir sample random lines
        gt_line_nums = set()
        reservoir = []  # (line_num,) for reservoir sampling
        line_count = 0

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            next(f)  # skip header
            for line_num, line in enumerate(f):
                parts = line.split("\t", 1)  # only split first column for speed
                pid = parts[0]
                if pid in needed_pool_ids:
                    gt_line_nums.add(line_num)
                else:
                    # Reservoir sampling for random negatives
                    if len(reservoir) < POOL_SAMPLE_SIZE:
                        reservoir.append(line_num)
                    else:
                        j = random.randint(0, line_count)
                        if j < POOL_SAMPLE_SIZE:
                            reservoir[j] = line_num
                line_count += 1

        selected_lines = gt_line_nums | set(reservoir)
        print(f"    {src_name}: {line_count:,} total, {len(gt_line_nums):,} GT+, {len(reservoir):,} sampled → loading {len(selected_lines):,}")

        # Pass 2: Load only selected lines
        loaded = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            next(f)  # skip header
            for line_num, line in enumerate(f):
                if line_num in selected_lines:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 4:
                        rec = LightRecord(parts[0], parts[1], parts[2], parts[3], source=src_name)
                        pool_all[parts[0]] = rec
                        loaded += 1
        print(f"    {src_name}: loaded {loaded:,} records")

    print(f"  Total pool: {len(pool_all):,} records ({time.time()-t1:.1f}s)")

    # Build IDF on training corpus
    print("  Building IDF model...")
    t1 = time.time()
    idf = LightIDF()
    for rec in s1_all.values():
        idf.add_record(rec)
    for rec in pool_all.values():
        idf.add_record(rec)
    idf.finalize()
    print(f"    IDF built over {idf.total_docs:,} docs ({time.time()-t1:.1f}s)")

    # Build pool index
    print("  Building pool index...")
    t1 = time.time()
    pool_idx = build_pool_index(pool_all)
    print(f"    Index built ({time.time()-t1:.1f}s)")

    # Block + extract features + build labels
    print("  Blocking + feature extraction for training...")
    t1 = time.time()
    pair_keys: List[Tuple[str, str]] = []
    feature_rows: List[Dict[str, float]] = []
    labels: List[int] = []

    for i, (sid, s1_rec) in enumerate(s1_all.items()):
        gt_matches = set(gt.get(sid, []))
        cands = retrieve_candidates(s1_rec, pool_idx)

        # Ensure all ground truth positives are in candidate set
        for m in gt_matches:
            if m not in cands and m in pool_all:
                cands.append(m)

        cand_count = len(cands)
        for pid in cands:
            pool_rec = pool_all.get(pid)
            if pool_rec is None:
                continue
            feats = extract_pair_features(s1_rec, pool_rec, idf, cand_count=cand_count)
            pair_keys.append((sid, pid))
            feature_rows.append(feats)
            labels.append(1 if pid in gt_matches else 0)

        if (i + 1) % 10000 == 0:
            print(f"    Processed {i+1:,}/{len(s1_all):,} S1 entities...")

    print(f"  Total pairs: {len(pair_keys):,} ({sum(labels):,} positives, {len(labels)-sum(labels):,} negatives) ({time.time()-t1:.1f}s)")

    # Build DataFrame
    df = pd.DataFrame(feature_rows)
    feature_cols = sorted(df.columns.tolist())
    df = df[feature_cols]
    labels_arr = np.array(labels, dtype=np.int32)

    # Free memory
    del feature_rows, pool_idx
    gc.collect()

    # Train 5-fold GroupKFold LightGBM
    print("\n  Training 5-fold LightGBM...")
    t1 = time.time()

    s1_ids_per_pair = [k[0] for k in pair_keys]
    unique_s1 = sorted(set(s1_ids_per_pair))
    np.random.shuffle(unique_s1)
    n_splits = 5
    fold_size = len(unique_s1) // n_splits
    folds = []
    for f_idx in range(n_splits):
        start = f_idx * fold_size
        end = start + fold_size if f_idx < n_splits - 1 else len(unique_s1)
        val_s1 = set(unique_s1[start:end])
        train_s1 = set(unique_s1) - val_s1
        folds.append((train_s1, val_s1))

    # Hard negative weights
    weights = np.ones(len(labels_arr), dtype=np.float32)
    for i in range(len(labels_arr)):
        if labels_arr[i] == 0:
            row = df.iloc[i]
            if row.get("name_hi_addr_lo", 0.0) == 1.0:
                weights[i] = 2.0
            elif row.get("house_conflict", 0.0) == 1.0:
                weights[i] = 2.0
            elif row.get("name_exact_fold", 0.0) == 1.0:
                weights[i] = 1.5

    # Monotone constraints
    MONOTONE_SPECS = {
        "name_exact_norm": 1, "name_exact_core": 1, "name_exact_fold": 1,
        "name_lev_ratio_core": 1, "name_jw_core": 1, "name_token_set_ratio_core": 1,
        "name_soft_tfidf": 1, "idf_shared_max": 1, "idf_shared_sum": 1,
        "acronym_match": 1, "legal_form_eq": 1, "alt_name_max_sim": 1,
        "legal_form_conflict": -1, "idf_unshared_max": -1, "digits_in_name_conflict": -1,
        "addr_exact_eq": 1, "addr_token_set_ratio": 1, "postal_eq": 1,
        "house_eq": 1, "unit_eq": 1, "nums_jaccard": 1, "addr_idf_shared_max": 1,
        "house_conflict": -1, "unit_conflict": -1, "postal_conflict": -1,
        "addr_idf_unshared_sum": -1, "same_country": 1, "name_sim_x_addr_sim": 1,
        "name_hi_addr_lo": -1,
    }
    monotone = [MONOTONE_SPECS.get(col, 0) for col in feature_cols]

    models: List[lgb.Booster] = []
    oof_preds = np.zeros(len(labels_arr), dtype=np.float64)

    for fold_idx, (train_s1, val_s1) in enumerate(folds):
        train_mask = np.array([sid in train_s1 for sid in s1_ids_per_pair])
        val_mask = np.array([sid in val_s1 for sid in s1_ids_per_pair])

        x_train = df.iloc[train_mask]
        y_train = labels_arr[train_mask]
        w_train = weights[train_mask]
        x_val = df.iloc[val_mask]
        y_val = labels_arr[val_mask]

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_data_in_leaf": max(20, int(len(x_train) * 0.001)),
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "max_bin": 255,
            "seed": seed + fold_idx,
            "deterministic": True,
            "verbosity": -1,
            "n_jobs": -1,
            "monotone_constraints": monotone,
            "monotone_constraints_method": "advanced",
        }

        dtrain = lgb.Dataset(x_train, label=y_train, weight=w_train, free_raw_data=False)
        dval = lgb.Dataset(x_val, label=y_val, reference=dtrain, free_raw_data=False)

        booster = lgb.train(
            params, dtrain,
            num_boost_round=3000,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)],
        )
        models.append(booster)
        oof_preds[val_mask] = booster.predict(x_val)
        print(f"    Fold {fold_idx+1}: best_iteration={booster.best_iteration}")

    # Isotonic calibration
    from sklearn.isotonic import IsotonicRegression
    calibrator = IsotonicRegression(y_min=1e-5, y_max=1.0 - 1e-5, increasing=True, out_of_bounds="clip")
    calibrator.fit(oof_preds, labels_arr)
    cal_oof = calibrator.predict(np.clip(oof_preds, 0, 1))

    # Tune decision parameters on OOF
    print("  Tuning decision parameters on OOF...")
    oof_pair_probs = {k: float(p) for k, p in zip(pair_keys, cal_oof)}
    excl_probs = apply_exclusivity(oof_pair_probs, delta=0.10, shrink=0.25, hard=False)

    s1_to_cands: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for (sid, pid), p in excl_probs.items():
        s1_to_cands[sid].append((pid, p))

    best_f05 = -1.0
    best_lam = 0.01
    best_beta = 0.0
    from ber.evaluation.f05 import compute_macro_f05

    for test_lam in [0.005, 0.01, 0.02, 0.03, 0.05]:
        for beta in np.linspace(-1.0, 1.0, 21):
            preds: Dict[str, List[str]] = {}
            for sid in sampled_s1_ids:
                cands_list = s1_to_cands.get(sid, [])
                if not cands_list:
                    preds[sid] = []
                    continue
                cands_sorted = sorted(cands_list, key=lambda x: -x[1])
                c_ids = [c[0] for c in cands_sorted]
                c_probs = [c[1] for c in cands_sorted]
                sel_idx, _, _ = select_expected_f05(c_probs, lam=test_lam, beta=float(beta))
                preds[sid] = [c_ids[i] for i in sel_idx]

            sampled_gt = {sid: gt[sid] for sid in sampled_s1_ids}
            result = compute_macro_f05(preds, sampled_gt)
            if result["macro_f05"] > best_f05:
                best_f05 = result["macro_f05"]
                best_lam = test_lam
                best_beta = float(beta)

    print(f"  Best OOF Macro F0.5: {best_f05:.4f} (lam={best_lam}, beta={best_beta:.2f})")
    print(f"  Training completed in {time.time()-t0:.1f}s total")

    # Save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(os.path.join(MODEL_DIR, f"model_fold{i}.txt"))
    with open(os.path.join(MODEL_DIR, "calibrator.pkl"), "wb") as f:
        pickle.dump(calibrator, f)
    with open(os.path.join(MODEL_DIR, "params.pkl"), "wb") as f:
        pickle.dump({
            "feature_cols": feature_cols,
            "best_lam": best_lam,
            "best_beta": best_beta,
            "best_f05": best_f05,
        }, f)

    return models, feature_cols, idf, {
        "calibrator": calibrator,
        "best_lam": best_lam,
        "best_beta": best_beta,
    }


# ────────────────────────────────────────────────────────────────
# Test inference
# ────────────────────────────────────────────────────────────────
def run_full_test():
    t_start = time.time()
    print("=" * 70)
    print("CPFC INFERENCE PIPELINE - REAL DATA")
    print("=" * 70)

    # Phase 1: Train model (or load if already trained)
    if os.path.exists(os.path.join(MODEL_DIR, "params.pkl")):
        print("\n[Phase 1] Loading pre-trained models...")
        models = []
        for i in range(5):
            m_path = os.path.join(MODEL_DIR, f"model_fold{i}.txt")
            if os.path.exists(m_path):
                models.append(lgb.Booster(model_file=m_path))
        with open(os.path.join(MODEL_DIR, "calibrator.pkl"), "rb") as f:
            calibrator = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "params.pkl"), "rb") as f:
            params = pickle.load(f)
        feature_cols = params["feature_cols"]
        optimal_lam = params["best_lam"]
        optimal_beta = params["best_beta"]
        print(f"  Loaded {len(models)} models, lam={optimal_lam}, beta={optimal_beta:.2f}")

        # We need IDF for test - build from test data
        idf = None  # Will be built from test data below
    else:
        print("\n[Phase 1] Training model on real data...")
        models, feature_cols, train_idf, opt_params = train_model(TRAIN_DIR)
        calibrator = opt_params["calibrator"]
        optimal_lam = opt_params["best_lam"]
        optimal_beta = opt_params["best_beta"]
        idf = None  # Will rebuild from test data

    # Phase 2: Index test pool
    print(f"\n[Phase 2] Loading and indexing test pool...")
    t0 = time.time()
    pool_records: Dict[str, LightRecord] = {}

    for src_num, src_name in [(2, "S2"), (3, "S3")]:
        path = os.path.join(TEST_DIR, f"test_source{src_num}.tsv")
        count = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4:
                    rec = LightRecord(parts[0], parts[1], parts[2], parts[3], source=src_name)
                    pool_records[parts[0]] = rec
                    count += 1
        print(f"  {src_name}: {count:,} records")

    print(f"  Total pool: {len(pool_records):,} records ({time.time()-t0:.1f}s)")

    # Build IDF from test data
    print("  Building test IDF...")
    t0 = time.time()
    idf = LightIDF()
    for rec in pool_records.values():
        idf.add_record(rec)

    # Also scan S1 for IDF (streamed)
    s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    s1_count = 0
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                tmp_rec = LightRecord(parts[0], parts[1], parts[2], parts[3])
                idf.add_record(tmp_rec)
                s1_count += 1
    idf.finalize()
    print(f"  IDF built over {idf.total_docs:,} docs ({time.time()-t0:.1f}s)")

    # Build pool index
    print("  Building pool inverted index...")
    t0 = time.time()
    pool_idx = build_pool_index(pool_records)
    print(f"  Index built ({time.time()-t0:.1f}s)")

    # Phase 3: Stream S1 entities → block → extract features → score → decide
    print(f"\n[Phase 3] Scoring {s1_count:,} test S1 entities...")
    t0 = time.time()

    s1_candidates: Dict[str, List[str]] = {}
    s1_matches: Dict[str, List[str]] = {}
    s1_pair_scores: Dict[Tuple[str, str], float] = {}

    s1_processed = 0
    batch_features: List[Dict[str, float]] = []
    batch_pair_keys: List[Tuple[str, str]] = []
    batch_s1_ids: List[str] = []
    batch_cands_map: Dict[str, List[str]] = {}
    BATCH_SIZE = 5000

    def flush_batch():
        """Score accumulated batch with the trained model and make decisions."""
        nonlocal batch_features, batch_pair_keys, batch_s1_ids, batch_cands_map

        if not batch_features:
            return

        df_batch = pd.DataFrame(batch_features)
        # Ensure all feature columns are present
        for col in feature_cols:
            if col not in df_batch.columns:
                df_batch[col] = 0.0
        df_batch = df_batch[feature_cols]

        # Ensemble prediction
        preds = np.zeros(len(batch_pair_keys), dtype=np.float64)
        for m in models:
            preds += m.predict(df_batch)
        preds /= len(models)

        # Calibrate
        cal_preds = calibrator.predict(np.clip(preds, 0.0, 1.0))

        # Build pair probs for this batch
        batch_pair_probs = {k: float(p) for k, p in zip(batch_pair_keys, cal_preds)}

        # Apply exclusivity
        excl_probs = apply_exclusivity(batch_pair_probs, delta=0.10, shrink=0.25, hard=False)

        # Group by S1
        s1_to_cands: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for (sid, pid), p in excl_probs.items():
            s1_to_cands[sid].append((pid, p))

        # Expected-F0.5 selection
        for sid in set(batch_s1_ids):
            cands_scored = s1_to_cands.get(sid, [])
            if not cands_scored:
                s1_matches[sid] = []
                continue
            cands_sorted = sorted(cands_scored, key=lambda x: -x[1])
            c_ids = [c[0] for c in cands_sorted]
            c_probs = [c[1] for c in cands_sorted]

            sel_indices, _, _ = select_expected_f05(
                c_probs, lam=optimal_lam, beta=optimal_beta, kmax=10
            )
            s1_matches[sid] = [c_ids[i] for i in sel_indices]

        # Clear batch
        batch_features = []
        batch_pair_keys = []
        batch_s1_ids = []
        batch_cands_map = {}

    # Stream through S1
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue

            sid = parts[0]
            s1_rec = LightRecord(sid, parts[1], parts[2], parts[3], source="S1")
            s1_processed += 1

            # Block
            cands = retrieve_candidates(s1_rec, pool_idx)
            s1_candidates[sid] = cands

            if not cands:
                s1_matches[sid] = []
                continue

            # Extract features for all candidates
            cand_count = len(cands)
            for pid in cands:
                pool_rec = pool_records.get(pid)
                if pool_rec is None:
                    continue
                feats = extract_pair_features(s1_rec, pool_rec, idf, cand_count=cand_count)
                batch_features.append(feats)
                batch_pair_keys.append((sid, pid))
                batch_s1_ids.append(sid)

            batch_cands_map[sid] = cands

            # Flush when batch is large enough
            if len(set(batch_s1_ids)) >= BATCH_SIZE:
                flush_batch()

            if s1_processed % 50000 == 0:
                elapsed = time.time() - t0
                rate = s1_processed / elapsed
                eta = (s1_count - s1_processed) / rate / 60
                print(f"    {s1_processed:,}/{s1_count:,} ({elapsed:.0f}s, {rate:.0f}/s, ETA {eta:.1f}min)")

    # Flush remaining
    flush_batch()

    # Ensure every S1 has an entry
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            sid = line.split("\t")[0]
            if sid not in s1_matches:
                s1_matches[sid] = []
            if sid not in s1_candidates:
                s1_candidates[sid] = []

    non_empty = sum(1 for m in s1_matches.values() if m)
    total_links = sum(len(m) for m in s1_matches.values())
    singleton_rate = 1.0 - non_empty / max(1, s1_processed)
    print(f"\n  Completed scoring in {time.time()-t0:.1f}s")
    print(f"  Non-empty: {non_empty:,} ({non_empty/max(1,s1_processed)*100:.1f}%)")
    print(f"  Singletons: {s1_processed - non_empty:,} ({singleton_rate*100:.1f}%)")
    print(f"  Total links: {total_links:,} (avg {total_links/max(1,non_empty):.2f} per non-singleton)")

    # Phase 4: Write submission
    print(f"\n[Phase 4] Writing submission files...")
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    matching_tsv = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_tsv = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    with open(matching_tsv, "w", encoding="utf-8", newline="") as f_m, \
         open(candidate_tsv, "w", encoding="utf-8", newline="") as f_c:

        w_m = csv.writer(f_m, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar=None, lineterminator="\n")
        w_c = csv.writer(f_c, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar=None, lineterminator="\n")

        w_m.writerow(["source1_entity_id", "matched_entity_ids"])
        w_c.writerow(["source1_entity_id", "candidate_entity_ids"])

        with open(s1_path, "r", encoding="utf-8", errors="replace") as f_s1:
            next(f_s1)
            for line in f_s1:
                sid = line.split("\t")[0]
                m_list = s1_matches.get(sid, [])
                c_list = s1_candidates.get(sid, [])
                w_m.writerow([sid, ",".join(m_list)])
                w_c.writerow([sid, ",".join(c_list)])

    print(f"  Wrote submission in {time.time()-t0:.1f}s")

    # Phase 5: Validate
    print("\n[Phase 5] Validating submission...")
    import subprocess
    cmd = [
        sys.executable,
        "student_resource/utils/validate_submission.py",
        "--matching", matching_tsv,
        "--candidate", candidate_tsv,
        "--test-dir", TEST_DIR,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    total_time = time.time() - t_start
    print(f"\nPipeline completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")


if __name__ == "__main__":
    run_full_test()
