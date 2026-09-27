# Calibrated Precision-First Cascade (CPFC) for Business Entity Resolution
### Amazon ML Challenge 2026 Solution

This repository contains the competition-grade solution for the Amazon ML Challenge Business Entity Resolution task, targeting macro-averaged per-entity $F_{0.5}$ with singleton rewards.

---

## 1. Quickstart

### Environment Setup
```bash
# Setup virtual environment with python 3.11
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Running Unit Tests
```bash
pytest tests/ -v
```

### End-to-End Pipeline Execution
```bash
# 1. Generate realistic benchmark data (if running locally without competition data)
python -m ber.cli generate-synthetic --train-size 100 --test-size 50

# 2. Run end-to-end CPFC cascade (EDA -> Blocking -> Features -> 5-Fold LGBM -> Calibration -> Expected-F0.5 Selection -> Submission Output)
python -m ber.cli run --config configs/final.yaml

# 3. Validate submission contract
python ../../utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir ../../dataset/test
```

---

## 2. Solution Architecture

The **Calibrated Precision-First Cascade (CPFC)** is structured into 9 modular phases:
1. **Safe TSV Ingestion (`ber.io.read_tsv`)**: Strict parsing preventing NaN conversion of strings like `"NA"`, `"None"` and quote corruption.
2. **EDA Decision Gates (`ber.validation.gt_checks`)**: Automated verification of Gates G1–G6 (exclusivity, country partitioning, postal prevalence, multi-match frequency).
3. **Multi-Representation Normalization (`ber.normalization`)**:
   - NFKC casefolding, accent stripping (stdlib `unicodedata`), transliteration folding (Table 10.3), acronym extraction.
   - Address normalization: street abbreviations, house/unit number extraction, postal code canonicalization, landmark isolation.
   - Open-set country normalization (Rule O8 compliant).
4. **Multi-Channel Candidate Blocking (`ber.blocking`)**:
   - C1: Exact hash keys (`name_core||country`, `name_core||postal`, `name_fold||country`)
   - C2: Sparse character (2,4)-gram TF-IDF cosine top-K per source
   - C3: Rare-token inverted index ($df \le 50$, ranked by $\sum \text{IDF}$)
   - C4: Phonetic folding and acronym indexing
   - C5: Address and numeric identifier keys
   - C6: Sorted Neighbourhood Method ($w=10$)
   - S2↔S3 candidate expansion
   - Budgeted candidate pruning to top-M while preserving multi-channel pairs.
5. **Feature Engineering (`ber.features`)**:
   - ~150 features across name similarities, address similarities, rare-token IDF statistics, cross-field interactions, and provenance masks.
6. **Primary Matcher (`ber.models.lgbm`)**:
   - 5-fold `GroupKFold` by Source 1 entity.
   - Directional monotone constraints (+1 on similarity features, -1 on branch conflicts like `house_conflict`).
   - Hard negative mining sample weights (HN1–HN7).
7. **Collective Stage 2b Matcher (`ber.models.stage2b`)**:
   - Incorporates out-of-fold group-context features (rank in S1, score margin to best, reverse competition rank, group entropy).
8. **Cross-Fitted Isotonic Calibration (`ber.calibration.isotonic`)**:
   - Maps model scores to true empirical probabilities.
9. **Decision Layer (`ber.decision`)**:
   - Soft exclusivity resolution.
   - Dynamic programming Expected-$F_{0.5}$ set selection (Appendix A) with $k=0$ singleton abstention.

---

## 3. Directory Layout
```
business_entity_resolution/
├── Makefile
├── requirements.txt
├── pyproject.toml
├── README.md
├── configs/
│   ├── base.yaml
│   ├── features.yaml
│   └── final.yaml
├── src/ber/
│   ├── io/
│   ├── validation/
│   ├── normalization/
│   ├── extraction/
│   ├── blocking/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── calibration/
│   ├── decision/
│   ├── evaluation/
│   ├── inference/
│   ├── data/
│   └── cli.py
└── tests/
    ├── test_f05.py
    ├── test_io.py
    ├── test_normalization.py
    ├── test_blocking.py
    ├── test_decision.py
    └── test_submission_contract.py
```
