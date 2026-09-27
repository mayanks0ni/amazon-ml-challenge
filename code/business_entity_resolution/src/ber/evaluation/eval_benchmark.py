"""Fast benchmark evaluation script to compute exact Macro F0.5 and metrics on real ground truth."""
import csv
import os
import re
import sys
import time
from collections import defaultdict

from ber.normalization.unicode import normalize_unicode
from ber.normalization.fold import fold_text
from ber.extraction.legal_form import extract_legal_form
from ber.extraction.postal import extract_postal_code
from ber.extraction.numbers import extract_address_numbers
from ber.evaluation.f05 import compute_macro_f05


def run_eval(n_entities: int = 10000, data_dir: str = "student_resource/dataset/train"):
    print("=" * 65)
    print(f"BENCHMARKING UPDATED PIPELINE ON {n_entities:,} GROUND-TRUTH ENTITIES")
    print("=" * 65)

    t0 = time.time()
    # 1. Load ground truth
    gt = {}
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    with open(gt_path, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if len(gt) >= n_entities:
                break
            gt[row[0]] = [x.strip() for x in row[1].split(",") if x.strip()]

    # 2. Load S1 records
    s1_data = {}
    with open(os.path.join(data_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if row[0] in gt:
                s1_data[row[0]] = (row[1], row[2], row[3].strip())

    # 3. Load Pool records (needed + 50k distractors)
    needed_pids = set()
    for m in gt.values():
        needed_pids.update(m)

    p_data = {}
    distractors = 0
    for s in (2, 3):
        with open(os.path.join(data_dir, f"train_source{s}.tsv"), "r", encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            next(r)
            for row in r:
                pid = row[0]
                if pid in needed_pids:
                    p_data[pid] = (row[1], row[2], row[3].strip())
                elif distractors < 50000:
                    p_data[pid] = (row[1], row[2], row[3].strip())
                    distractors += 1

    print(f"Loaded {len(s1_data):,} S1 and {len(p_data):,} Pool records in {time.time() - t0:.2f}s.")

    # 4. Inverted index (matching run_full_test_inference.py with wider blocking)
    t0 = time.time()
    index_core_country = defaultdict(list)
    index_clean_country = defaultdict(list)
    index_postal_house = defaultdict(list)
    index_clean_addr = defaultdict(list)
    index_sorted_addr = defaultdict(list)
    index_fold_country = defaultdict(list)
    index_token = defaultdict(list)

    addr_stop = {
        "st", "street", "ave", "avenue", "rd", "road", "blvd", "dr", "drive",
        "ln", "lane", "ct", "court", "pl", "place", "terrace", "null", "township",
        "apt", "suite", "ste", "flr", "floor"
    }

    for pid, (name, addr, c) in p_data.items():
        norm_name = normalize_unicode(name)
        norm_name = re.sub(r"https?://", "", norm_name)
        norm_name = re.sub(r"\bwww\.", "", norm_name)
        norm_name = re.sub(r"\.(?:com|org|net|co\.in|in|co|io|biz|info|gov|edu|fr|us)(?=[/\s]|$)", "", norm_name).strip()

        core, _ = extract_legal_form(norm_name)
        if core:
            postings = index_core_country[(core, c)]
            if len(postings) < 50: postings.append(pid)

            fold = fold_text(core)
            if fold and fold != core:
                fold_p = index_fold_country[(fold, c)]
                if len(fold_p) < 50: fold_p.append(pid)

            clean_n = "".join(ch for ch in core if ch.isalnum())
            if clean_n:
                cn_p = index_clean_country[(clean_n, c)]
                if len(cn_p) < 50: cn_p.append(pid)

            # Token-level index
            core_tokens = core.split()
            for t in core_tokens:
                if len(t) >= 2:
                    t_p = index_token[(t, c)]
                    if len(t_p) < 500: t_p.append(pid)

        postal = extract_postal_code(addr, country=c)
        _, house_no, _ = extract_address_numbers(addr)
        if postal and house_no:
            loc_p = index_postal_house[(postal, house_no, c)]
            if len(loc_p) < 15: loc_p.append(pid)

        c_addr = "".join(ch for ch in addr.lower() if ch.isalnum())
        if len(c_addr) >= 10:
            ca_p = index_clean_addr[(c_addr[:25], c)]
            if len(ca_p) < 20: ca_p.append(pid)

        s_ord = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", addr.lower())
        s_toks = re.findall(r"[a-z0-9]+", s_ord)
        meaningful = sorted([t for t in s_toks if t not in addr_stop])
        s_addr_key = "".join(meaningful[:5])
        if len(s_addr_key) >= 6:
            sa_p = index_sorted_addr[(s_addr_key, c)]
            if len(sa_p) < 20: sa_p.append(pid)

    # 5. Query and match
    predictions = {}
    candidates = {}
    matched_pool_ids = set()

    for sid, (name, addr, c) in s1_data.items():
        norm_name = normalize_unicode(name)
        norm_name = re.sub(r"https?://", "", norm_name)
        norm_name = re.sub(r"\bwww\.", "", norm_name)
        norm_name = re.sub(r"\.(?:com|org|net|co\.in|in|co|io|biz|info|gov|edu|fr|us)(?=[/\s]|$)", "", norm_name).strip()

        core, _ = extract_legal_form(norm_name)
        cands = []
        seen = set()

        if core:
            for pid in index_core_country.get((core, c), []):
                if pid not in seen: seen.add(pid); cands.append(pid)
            fold = fold_text(core)
            for pid in index_fold_country.get((fold, c), []):
                if pid not in seen: seen.add(pid); cands.append(pid)
            clean_n = "".join(ch for ch in core if ch.isalnum())
            if clean_n:
                for pid in index_clean_country.get((clean_n, c), []):
                    if pid not in seen: seen.add(pid); cands.append(pid)

            # Token-overlap retrieval
            core_tokens = core.split()
            if len(core_tokens) >= 2:
                token_scores = defaultdict(int)
                for t in core_tokens:
                    if len(t) >= 2:
                        for pid in index_token.get((t, c), []):
                            if pid not in seen:
                                token_scores[pid] += 1
                overlap_cands = sorted(
                    ((pid, sc) for pid, sc in token_scores.items() if sc >= 2),
                    key=lambda x: -x[1]
                )
                for pid, _ in overlap_cands[:30]:
                    seen.add(pid); cands.append(pid)

        c_addr = "".join(ch for ch in addr.lower() if ch.isalnum())
        if len(c_addr) >= 10:
            for pid in index_clean_addr.get((c_addr[:25], c), []):
                if pid not in seen: seen.add(pid); cands.append(pid)

        postal = extract_postal_code(addr, country=c)
        _, house_no, _ = extract_address_numbers(addr)
        if postal and house_no:
            for pid in index_postal_house.get((postal, house_no, c), []):
                if pid not in seen: seen.add(pid); cands.append(pid)

        s_ord = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", addr.lower())
        s_toks = re.findall(r"[a-z0-9]+", s_ord)
        meaningful = sorted([t for t in s_toks if t not in addr_stop])
        s_addr_key = "".join(meaningful[:5])
        if len(s_addr_key) >= 6:
            for pid in index_sorted_addr.get((s_addr_key, c), []):
                if pid not in seen: seen.add(pid); cands.append(pid)

        capped = cands[:60]
        candidates[sid] = capped

        matches = []
        for pid in capped:
            if pid not in matched_pool_ids:
                matched_pool_ids.add(pid)
                matches.append(pid)
        predictions[sid] = matches

    # 6. Compute metrics
    metrics = compute_macro_f05(predictions, gt)

    total_tp = 0
    total_pred = 0
    total_gold = sum(len(v) for v in gt.values())
    total_candidates_covered = 0
    total_blocking_failures = 0

    for sid in gt:
        p_set = set(predictions.get(sid, []))
        c_set = set(candidates.get(sid, []))
        g_set = set(gt[sid])
        total_tp += len(p_set & g_set)
        total_pred += len(p_set)
        total_candidates_covered += len(c_set & g_set)
        total_blocking_failures += len(g_set - c_set)

    micro_prec = total_tp / max(1, total_pred)
    micro_rec = total_tp / max(1, total_gold)
    cand_pc = total_candidates_covered / max(1, total_gold)

    false_merges = total_pred - total_tp
    missed_in_cands = total_candidates_covered - total_tp

    singletons_gt = [sid for sid in gt if len(gt[sid]) == 0]
    singletons_correct = sum(1 for sid in singletons_gt if len(predictions.get(sid, [])) == 0)
    singleton_acc = singletons_correct / max(1, len(singletons_gt)) if singletons_gt else 1.0

    # Compute blocking ceiling F0.5 (what score would be if model were perfect on candidates)
    ceiling_preds = {}
    for sid in gt:
        c_set = set(candidates.get(sid, []))
        g_set = set(gt[sid])
        ceiling_preds[sid] = list(c_set & g_set)
    ceiling_metrics = compute_macro_f05(ceiling_preds, gt)

    print("=" * 65)
    print("UPDATED PIPELINE METRICS REPORT")
    print("=" * 65)
    print(f"  >> Macro F0.5 Score:             {metrics['macro_f05']:.5f}")
    print(f"  >> Non-Singleton F0.5 Score:     {metrics['non_singleton_f05']:.5f}")
    print(f"  >> Micro Precision:              {micro_prec:.5f} ({total_tp:,} true / {total_pred:,} predicted)")
    print(f"  >> Micro Recall:                 {micro_rec:.5f} ({total_tp:,} true / {total_gold:,} gold)")
    print(f"  >> Singleton Accuracy:           {singleton_acc:.5f} ({singletons_correct:,}/{len(singletons_gt):,} correct)")
    print(f"  >> Blocking Pair Completeness:   {cand_pc:.5f}")
    print(f"  >> Blocking Ceiling F0.5:        {ceiling_metrics['macro_f05']:.5f}")
    print(f"  >> False Merges (False Pos):     {false_merges}")
    print(f"  >> Missed in Candidates:         {missed_in_cands}")
    print(f"  >> Blocking Failures:            {total_blocking_failures}")
    print(f"  >> Total Benchmark Runtime:      {time.time() - t0:.2f}s")
    print("=" * 65)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    run_eval(n)
