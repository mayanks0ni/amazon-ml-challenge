"""
ber_lib.py - Amazon ML Challenge: Business Entity Resolution
Calibrated Precision-First Cascade (CPFC), memory-safe edition for Google Colab.

Stages (each saves its output under WORK_DIR so the pipeline can resume after a disconnect):
  step2_normalize        raw TSV -> normalized parquet (names, addresses, keys)
  step3_baseline         exact-match baseline + train F0.5 + baseline submission
  step4_train_candidates multi-channel blocking for a sample of train S1 + recall report
  step5_train_features   pair features for the train candidates
  step6_train_stage2a    LightGBM pair model, GroupKFold by S1 -> OOF probabilities
  step7_train_stage2b    group-context LightGBM on OOF probabilities
  step8_decide           isotonic calibration + decision-rule tuning (per-entity F0.5)
  step9_error_analysis   error buckets on out-of-fold predictions
  step11_predict_test    full test inference -> matching_results.tsv, candidate_pairs.tsv
  package_submission     output/ + code/ + docs -> zip
Only MIT/BSD/Apache libraries are used; no external data, no pretrained models.
"""
import os, gc, re, sys, csv, json, math, time, glob, shutil, zipfile, unicodedata, subprocess
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

# ============================================================ config / logging
DEFAULT_CFG = dict(
    workers=max(1, os.cpu_count() or 1),
    train_s1_sample=150_000,   # train S1 entities used to train the models
    k_name=20, k_addr=20,      # candidates kept per S1 from the name / address channel
    k_joint=20,                # joint name+address channel (0 disables it)
    df_cap_name=1000, df_cap_addr=1000, df_cap_joint=1000,   # keys shared by more pool records than this are ignored
    max_keys_name=15, max_keys_addr=25, max_keys_joint=40,   # rarest keys used per query record
    max_pairs_block=12_000_000,           # memory budget of one blocking chunk (pairs)
    feat_chunk=1_500_000,                 # pairs per feature chunk
    n_folds=5, seed=42,
    lgb_rounds=3000, lgb_lr=0.05, early_stop=100,
    kmax=20,
)
T0 = time.time()

def log(msg):
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram = f'RAM {vm.used / 1e9:4.1f}/{vm.total / 1e9:.1f} GB'
    except Exception:
        ram = ''
    print(f'[{time.time() - T0:7.1f}s | {ram}] {msg}', flush=True)

def cfg_with(cfg=None):
    c = dict(DEFAULT_CFG)
    c.update(cfg or {})
    return c

# ============================================================ I/O
SOURCE_COLS = ['entity_id', 'business_name', 'business_address', 'country']
GT_COLS = ['source1_entity_id', 'matched_entity_ids']

def read_tsv(path, cols=SOURCE_COLS):
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, na_filter=False,
                     quoting=csv.QUOTE_NONE, encoding='utf-8-sig', on_bad_lines='warn')
    df.columns = [c.strip() for c in df.columns]
    assert df.columns.tolist() == cols, f'{path}: header {df.columns.tolist()} != {cols}'
    return df.fillna('')

def fp(values):
    """64-bit fingerprint of strings (compact ID comparison)."""
    return pd.util.hash_pandas_object(pd.Series(values), index=False).values

def load_gold(data_dir):
    gt = read_tsv(f'{data_dir}/train_ground_truth.tsv', GT_COLS)
    g = gt.assign(r=gt.matched_entity_ids.str.split(',')).explode('r')[['source1_entity_id', 'r']]
    g = g.rename(columns={'source1_entity_id': 's1'})
    g['r'] = g.r.str.strip()
    g = g[g.r.notna() & (g.r != '')].reset_index(drop=True)
    return gt[['source1_entity_id']].rename(columns={'source1_entity_id': 's1'}), g

# ============================================================ text normalization
_CLEAN = {}
for _cp in range(sys.maxunicode + 1):
    _c = unicodedata.category(chr(_cp))
    if _c[0] in 'PSZ' or _c == 'Cc':
        _CLEAN[_cp] = ' '
    elif _c == 'Cf':
        _CLEAN[_cp] = None

# ---- Indic scripts -> Latin, built from Unicode character names (no external data) ----
_VOW = {'A': 'a', 'AA': 'a', 'I': 'i', 'II': 'i', 'U': 'u', 'UU': 'u', 'E': 'e', 'EE': 'e', 'AI': 'ai',
        'O': 'o', 'OO': 'o', 'AU': 'au', 'VOCALIC R': 'ri', 'VOCALIC RR': 'ri', 'VOCALIC L': 'li',
        'VOCALIC LL': 'li', 'SHORT E': 'e', 'SHORT O': 'o', 'CANDRA E': 'e', 'CANDRA O': 'o', 'CANDRA A': 'a',
        'OE': 'o', 'OOE': 'o', 'UE': 'u', 'UUE': 'u', 'AW': 'au', 'PRISHTHAMATRA E': 'e'}
_CONS_FIX = {'tt': 't', 'tth': 'th', 'dd': 'd', 'ddh': 'dh', 'nn': 'n', 'nnn': 'n', 'ny': 'n', 'ng': 'n',
             'ss': 'sh', 'll': 'l', 'lll': 'zh', 'rr': 'r', 'c': 'ch', 'dddh': 'dh', 'khh': 'kh', 'ghh': 'gh',
             'yy': 'y', 'rh': 'r'}
_I_CONS, _I_VOW, _I_SIGN, _I_OTHER, _I_VIRAMA = {}, {}, {}, {}, set()
_I_IGNORE = {'\u200c', '\u200d'}
for _cp in range(0x0900, 0x0E00):
    _ch = chr(_cp)
    _nm = unicodedata.name(_ch, '')
    if ' ' not in _nm:
        continue
    _rest = _nm.split(' ', 1)[1]
    if _rest.startswith('LETTER '):
        _L = _rest[7:]
        if _L in _VOW:
            _I_VOW[_ch] = _VOW[_L]
        elif _L.startswith('CHILLU ') or _L == 'KHANDA TA':
            _r = _L.split(' ')[-1].lower()
            _r = _r[:-1] if _r.endswith('a') and len(_r) > 1 else _r
            _I_OTHER[_ch] = _CONS_FIX.get(_r, _r)
        elif ' ' not in _L:
            _r = _L.lower()
            _r = _r[:-1] if _r.endswith('a') and len(_r) > 1 else _r
            _I_CONS[_ch] = _CONS_FIX.get(_r, _r)
    elif _rest.startswith('VOWEL SIGN '):
        _v = _rest[11:]
        _I_SIGN[_ch] = _VOW.get(_v, re.sub('[^a-z]', '', _v.lower())[:2])
    elif 'VIRAMA' in _rest:
        _I_VIRAMA.add(_ch)
    elif 'ANUSVARA' in _rest or 'CANDRABINDU' in _rest:
        _I_OTHER[_ch] = 'n'
    elif 'VISARGA' in _rest:
        _I_OTHER[_ch] = 'h'
    elif _rest.startswith('DIGIT '):
        _I_OTHER[_ch] = str(unicodedata.digit(_ch))
    elif _rest.startswith('SIGN') or _rest.startswith('AU LENGTH') or _rest.startswith('AI LENGTH'):
        _I_IGNORE.add(_ch)
    elif _rest == 'OM':
        _I_OTHER[_ch] = 'om'
_INDIC_RX = re.compile('[\u0900-\u0DFF]')

def translit_indic(s):
    out, pending = [], False
    for ch in s:
        if ch in _I_CONS:
            if pending:
                out.append('a')
            out.append(_I_CONS[ch]); pending = True
        elif ch in _I_SIGN:
            out.append(_I_SIGN[ch]); pending = False
        elif ch in _I_VIRAMA:
            pending = False
        elif ch in _I_VOW:
            if pending:
                out.append('a')
            out.append(_I_VOW[ch]); pending = False
        elif ch in _I_OTHER:
            if pending:
                out.append('a')
            out.append(_I_OTHER[ch]); pending = False
        elif ch in _I_IGNORE:
            continue
        else:
            pending = False          # word end: inherent vowel dropped (schwa deletion)
            out.append(ch)
    return ''.join(out)

def _join_single_letters(toks):
    out, run = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
        else:
            if run:
                out.append(''.join(run)) if len(run) > 1 else out.extend(run)
                run = []
            out.append(t)
    if run:
        out.append(''.join(run)) if len(run) > 1 else out.extend(run)
    return out

def base_tokens(s):
    """NFKC + casefold + Indic transliteration + accent folding + punctuation removal."""
    s = unicodedata.normalize('NFKC', s).casefold()
    tr = False
    if not s.isascii():
        if _INDIC_RX.search(s):
            s = translit_indic(s); tr = True
        s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    s = s.replace('&', ' and ').translate(_CLEAN)
    return _join_single_letters(s.split()), tr

LEGAL = {'limited': 'ltd', 'ltd': 'ltd', 'limitet': 'ltd', 'limted': 'ltd', 'limitad': 'ltd', 'ltda': 'ltd',
         'private': 'pvt', 'pvt': 'pvt', 'pvtltd': 'pvt', 'praivet': 'pvt', 'prayvet': 'pvt', 'privet': 'pvt',
         'llc': 'llc', 'inc': 'inc', 'incorporated': 'inc', 'corp': 'corp', 'corporation': 'corp',
         'co': 'co', 'company': 'co', 'lp': 'lp', 'llp': 'llp', 'pllc': 'pllc', 'pc': 'pc', 'plc': 'plc',
         'sarl': 'sarl', 'sarlu': 'sarl', 'sas': 'sas', 'sasu': 'sas', 'sa': 'sa', 'eurl': 'eurl', 'snc': 'snc',
         'sci': 'sci', 'selarl': 'selarl', 'eirl': 'eirl', 'scop': 'scop', 'scm': 'scm', 'gie': 'gie',
         'cie': 'co', 'compagnie': 'co',
         'gmbh': 'gmbh', 'ag': 'ag', 'bv': 'bv', 'pty': 'pty', 'opc': 'opc', 'elaelapi': 'llp'}
NAME_STOP = {'the', 'and', 'of', 'a', 'an', 'ms', 'et', 'le', 'la', 'les', 'de', 'du', 'des',
             'l', 'd', 'au', 'aux', 'en'}   # French articles; "l'atelier" -> "l atelier"

_US_STATES = {'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca',
              'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de', 'florida': 'fl', 'georgia': 'ga',
              'hawaii': 'hi', 'idaho': 'id', 'illinois': 'il', 'indiana': 'in', 'iowa': 'ia', 'kansas': 'ks',
              'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md', 'massachusetts': 'ma',
              'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms', 'missouri': 'mo', 'montana': 'mt',
              'nebraska': 'ne', 'nevada': 'nv', 'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm',
              'new york': 'ny', 'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok',
              'oregon': 'or', 'pennsylvania': 'pa', 'rhode island': 'ri', 'south carolina': 'sc',
              'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut', 'vermont': 'vt',
              'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv', 'wisconsin': 'wi', 'wyoming': 'wy',
              'district of columbia': 'dc'}
_IN_STATES = {'andhra pradesh': 'ap', 'arunachal pradesh': 'ar', 'assam': 'as', 'bihar': 'br',
              'chhattisgarh': 'cg', 'goa': 'ga', 'gujarat': 'gj', 'haryana': 'hr', 'himachal pradesh': 'hp',
              'jharkhand': 'jh', 'karnataka': 'ka', 'kerala': 'kl', 'madhya pradesh': 'mp', 'maharashtra': 'mh',
              'manipur': 'mn', 'meghalaya': 'ml', 'mizoram': 'mz', 'nagaland': 'nl', 'odisha': 'od',
              'orissa': 'od', 'punjab': 'pb', 'rajasthan': 'rj', 'sikkim': 'sk', 'tamil nadu': 'tn',
              'telangana': 'tg', 'tripura': 'tr', 'uttar pradesh': 'up', 'uttarakhand': 'uk',
              'uttaranchal': 'uk', 'west bengal': 'wb', 'delhi': 'dl', 'jammu and kashmir': 'jk',
              'ladakh': 'la', 'chandigarh': 'ch', 'puducherry': 'py', 'pondicherry': 'py',
              'bombay': 'mumbai', 'bangalore': 'bengaluru', 'madras': 'chennai', 'calcutta': 'kolkata',
              'maharashtr': 'mh', 'telangan': 'tg', 'karnatak': 'ka', 'keralam': 'kl', 'gurgaon': 'gurugram', 'poona': 'pune', 'belgaum': 'belagavi'}
_MULTI = {**_US_STATES, **_IN_STATES}
_MULTI_RX = re.compile(r'\b(' + '|'.join(sorted(map(re.escape, _MULTI), key=len, reverse=True)) + r')\b')
_STREET = {'street': 'st', 'road': 'rd', 'drive': 'dr', 'avenue': 'ave', 'av': 'ave', 'lane': 'ln',
           'court': 'ct', 'boulevard': 'blvd', 'bd': 'blvd', 'place': 'pl', 'circle': 'cir', 'highway': 'hwy',
           'parkway': 'pkwy', 'terrace': 'ter', 'square': 'sq', 'suite': 'ste', 'apartment': 'apt',
           'building': 'bldg', 'floor': 'fl', 'flr': 'fl', 'number': 'no', 'near': 'nr', 'opposite': 'opp',
           'north': 'n', 'south': 's', 'east': 'e', 'west': 'w', 'ts': 'tg',
           # French street types / abbreviations (country-agnostic renames: applied to both sides alike)
           'r': 'rue', 'bld': 'blvd', 'chemin': 'chem', 'che': 'chem', 'impasse': 'imp', 'route': 'rte',
           'allees': 'allee', 'all': 'allee', 'faubourg': 'fbg', 'residence': 'res', 'quartier': 'qu',
           'saint': 'st', 'sainte': 'ste', 'bis': 'b'}
_ORD_RX = re.compile(r'^(\d+)(st|nd|rd|th)$')
_PH = [('ph', 'f'), ('bh', 'b'), ('dh', 'd'), ('th', 't'), ('kh', 'k'), ('gh', 'g'), ('jh', 'j'), ('sh', 's'),
       ('ch', 'c'), ('ck', 'k'), ('q', 'k'), ('x', 'ks'), ('w', 'v'), ('z', 'j'), ('c', 'k'), ('y', 'i')]
_REP_RX = re.compile(r'(.)\1+')

@lru_cache(maxsize=2 ** 20)
def phon_key(t):
    """Consonant-skeleton phonetic key: bridges spelling/transliteration variants."""
    if not t.isalpha():
        return t
    if t.startswith('yu'):
        t = t[1:]
    for a, b in _PH:
        t = t.replace(a, b)
    return _REP_RX.sub(r'\1', t[0] + ''.join(ch for ch in t[1:] if ch not in 'aeiou'))

def norm_name(s):
    toks, tr = base_tokens(s)
    legal = sorted({LEGAL[t] for t in toks if t in LEGAL})
    core = [t for t in toks if t not in LEGAL and t not in NAME_STOP] or [t for t in toks if t not in NAME_STOP] or toks
    return (' '.join(toks), ' '.join(core), ' '.join(phon_key(t) for t in core), ' '.join(legal), tr)

def norm_addr(s):
    toks, tr = base_tokens(s)
    toks = [(_ORD_RX.sub(r'\1', t) if t[0].isdigit() else t) for t in toks]
    j = _MULTI_RX.sub(lambda m: _MULTI[m.group(1)], ' '.join(toks))
    toks = [_STREET.get(t, t) for t in j.split()]
    nums = list(dict.fromkeys(t for t in toks if any(c.isdigit() for c in t)))
    house = next((t for t in toks if t[0].isdigit()), '')
    return (' '.join(toks), house, ' '.join(nums), tr)

def norm_country(s):
    toks, _ = base_tokens(s)
    return ' '.join(toks)

def _batch(fn, items):
    return [fn(x) for x in items]

def _pmap(fn_name, items, workers):
    fn = {'name': norm_name, 'addr': norm_addr, 'country': norm_country}[fn_name]
    if workers <= 1 or len(items) < 20000:
        return [fn(x) for x in items]
    n = max(1, len(items) // (workers * 8))
    chunks = [items[i:i + n] for i in range(0, len(items), n)]
    out = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_batch, [fn] * len(chunks), chunks):
            out.extend(r)
    return out

def normalize_frame(df, workers=1):
    out = pd.DataFrame({'entity_id': df.entity_id.values})
    codes, uniq = pd.factorize(df.country)
    out['country'] = np.array(_pmap('country', list(uniq), 1), dtype=object)[codes] if len(uniq) else ''
    codes, uniq = pd.factorize(df.business_name)
    res = _pmap('name', list(uniq), workers)
    for i, col in enumerate(['name_n', 'name_core', 'name_phon', 'legal']):
        out[col] = np.array([r[i] for r in res], dtype=object)[codes]
    out['name_tr'] = np.array([r[4] for r in res], dtype=bool)[codes]
    del res; gc.collect()
    codes, uniq = pd.factorize(df.business_address)
    res = _pmap('addr', list(uniq), workers)
    for i, col in enumerate(['addr_n', 'house', 'nums']):
        out[col] = np.array([r[i] for r in res], dtype=object)[codes]
    out['addr_tr'] = np.array([r[3] for r in res], dtype=bool)[codes]
    return out

# ============================================================ step 2
FILES = [('train', 1), ('train', 2), ('train', 3), ('test', 1), ('test', 2), ('test', 3)]

def norm_path(work_dir, split, k):
    return f'{work_dir}/norm_{split}_s{k}.parquet'

def step2_normalize(data_dir, work_dir, cfg=None, force=False):
    cfg = cfg_with(cfg)
    os.makedirs(work_dir, exist_ok=True)
    for split, k in FILES:
        path = norm_path(work_dir, split, k)
        if os.path.exists(path) and not force:
            log(f'{os.path.basename(path)} exists, skipped'); continue
        df = read_tsv(f'{data_dir}/{split}_source{k}.tsv')
        log(f'{split}_source{k}: {len(df):,} rows loaded, normalizing...')
        out = normalize_frame(df, cfg['workers'])
        del df; gc.collect()
        out.to_parquet(path + '.tmp', index=False)
        os.replace(path + '.tmp', path)
        log(f'saved {os.path.basename(path)}')
        del out; gc.collect()
    ex = pd.read_parquet(norm_path(work_dir, 'train', 2)).head(5)
    return ex

NORM_COLS = ['entity_id', 'country', 'name_n', 'name_core', 'name_phon', 'legal', 'name_tr',
             'addr_n', 'house', 'nums', 'addr_tr']

def load_norm(work_dir, split, k, cols=None, country=None):
    df = pd.read_parquet(norm_path(work_dir, split, k), columns=cols)
    if country is not None:
        df = df[df.country == country]
    return df

def load_pool(work_dir, split, country=None, cols=None):
    parts = []
    for k in (2, 3):
        d = load_norm(work_dir, split, k, cols, country)
        d['is_s3'] = np.int8(k == 3)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)

# ============================================================ scoring
def f05_per_entity(all_s1, pred, gold):
    """all_s1: array of S1 ids. pred/gold: DataFrames with columns s1, r. Returns Series of F0.5 per S1."""
    idx = pd.Index(pd.unique(np.asarray(all_s1)))
    P = pred.groupby('s1').size().reindex(idx, fill_value=0).values
    G = gold.groupby('s1').size().reindex(idx, fill_value=0).values
    tp = pred.merge(gold, on=['s1', 'r']).groupby('s1').size().reindex(idx, fill_value=0).values
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.where(G == 0, (P == 0).astype(float), 5 * tp / (4 * P + G))
    return pd.Series(f, index=idx)

# ============================================================ step 3: baseline
def step3_baseline(data_dir, work_dir, out_dir, cfg=None):
    from rapidfuzz import fuzz, process
    cfg = cfg_with(cfg)
    def predict(split):
        s1 = load_norm(work_dir, split, 1, ['entity_id', 'country', 'name_core', 'addr_n'])
        pool = load_pool(work_dir, split, cols=['entity_id', 'country', 'name_core', 'addr_n'])
        pairs = []
        for key in ('name_core', 'addr_n'):
            a = s1[s1[key] != ''][['entity_id', 'country', key] + (['addr_n'] if key == 'name_core' else ['name_core'])]
            b = pool[pool[key] != ''][['entity_id', 'country', key] + (['addr_n'] if key == 'name_core' else ['name_core'])]
            b = b[b.groupby(['country', key]).entity_id.transform('size') <= 11]
            m = a.merge(b, on=['country', key], suffixes=('_a', '_b'))
            m = m[m.groupby('entity_id_a').entity_id_b.transform('size') <= 11].reset_index(drop=True)
            if key == 'name_core':
                sim = process.cpdist(m.addr_n_a.values, m.addr_n_b.values, scorer=fuzz.token_set_ratio, workers=-1)
                keep = (sim >= 60) | (m.addr_n_b.values == '')
                m = m[keep].assign(score=1.0 + sim[keep] / 100)
            else:
                sim = process.cpdist(m.name_core_a.values, m.name_core_b.values, scorer=fuzz.token_set_ratio, workers=-1)
                m = m.assign(score=sim / 100)
            pairs.append(m[['entity_id_a', 'entity_id_b', 'score']])
            del a, b, m; gc.collect()
        p = pd.concat(pairs).sort_values('score', ascending=False)
        p = p.drop_duplicates(['entity_id_a', 'entity_id_b']).drop_duplicates('entity_id_b')   # exclusivity
        p = p.rename(columns={'entity_id_a': 's1', 'entity_id_b': 'r'})[['s1', 'r']]
        return s1.entity_id.values, p
    ids, p = predict('train')
    _, gold = load_gold(data_dir)
    f = f05_per_entity(ids, p, gold)
    log(f'BASELINE train macro F0.5 = {f.mean():.4f}  ({len(p):,} predicted pairs)')
    ids, p = predict('test')
    bdir = f'{out_dir}/baseline'
    os.makedirs(bdir, exist_ok=True)
    write_submission(bdir, ids, p, p)
    report = validate_outputs(bdir, work_dir, data_dir)
    return float(f.mean()), report

# ============================================================ blocking
def _explode(texts):
    t = pd.Series(texts).str.split().explode()
    t = t[t.notna()]
    return t.index.values.astype(np.int32), t.values

def make_keys(df, channel, chunk=400_000):
    """(row, key-fingerprint) for every blocking key of every record."""
    rows_all, keys_all = [], []
    for s in range(0, len(df), chunk):
        d = df.iloc[s:s + chunk]
        R, K = [], []
        if channel == 'name':
            r, t = _explode(d.name_core.values)
            keep = np.fromiter((len(x) >= 2 or x.isdigit() for x in t), bool, len(t))
            R.append(r[keep]); K.append('n' + t[keep])
            same = r[1:] == r[:-1]
            R.append(r[:-1][same]); K.append('b' + t[:-1][same] + '_' + t[1:][same])
            r2, t2 = _explode(d.name_phon.values)
            keep = np.fromiter((len(x) >= 3 for x in t2), bool, len(t2))
            R.append(r2[keep]); K.append('p' + t2[keep])
            core = d.name_core.values
            nz = np.flatnonzero(core != '')
            R.append(nz.astype(np.int32)); K.append('w' + core[nz])
        else:
            r, t = _explode(d.addr_n.values)
            keep = np.fromiter((len(x) >= 3 or any(c.isdigit() for c in x) for x in t), bool, len(t))
            R.append(r[keep]); K.append('a' + t[keep])
            same = r[1:] == r[:-1]
            R.append(r[:-1][same]); K.append('c' + t[:-1][same] + '_' + t[1:][same])
        r = np.concatenate(R) + s
        k = fp(np.concatenate(K))
        u = pd.DataFrame({'r': r, 'k': k}).drop_duplicates()
        rows_all.append(u.r.values.astype(np.int32)); keys_all.append(u.k.values)
    if not rows_all:
        return np.zeros(0, np.int32), np.zeros(0, np.uint64)
    return np.concatenate(rows_all), np.concatenate(keys_all)

def block_channel(q_rows, q_keys, p_rows, p_keys, n_pool, K, df_cap, max_keys, max_pairs):
    """Top-K pool records per query row by summed IDF of shared rare keys."""
    empty = pd.DataFrame({'q': np.zeros(0, np.int32), 'p': np.zeros(0, np.int32),
                          'score': np.zeros(0, np.float32), 'rank': np.zeros(0, np.int16)})
    if len(q_keys) == 0 or len(p_keys) == 0:
        return empty
    uk, inv, cnt = np.unique(p_keys, return_inverse=True, return_counts=True)
    keep = cnt[inv] <= df_cap
    p_keys, p_rows = p_keys[keep], p_rows[keep]
    del inv, keep; gc.collect()
    o = np.argsort(p_keys, kind='stable')
    pk, pr = p_keys[o], p_rows[o]
    del o; gc.collect()
    uk, first, cnt = np.unique(pk, return_index=True, return_counts=True)
    if len(uk) == 0:
        return empty
    pos = np.searchsorted(uk, q_keys)
    pos[pos >= len(uk)] = 0
    ok = uk[pos] == q_keys
    q_rows, pos = q_rows[ok], pos[ok]
    df = cnt[pos]
    w = np.log((n_pool + 1) / df).astype(np.float32)
    o = np.lexsort((df, q_rows))                   # rarest keys first within each query
    q_rows, pos, df, w = q_rows[o], pos[o], df[o], w[o]
    if len(q_rows) == 0:
        return empty
    starts = np.r_[0, np.flatnonzero(np.diff(q_rows)) + 1]
    rank = np.arange(len(q_rows)) - np.repeat(starts, np.diff(np.r_[starts, len(q_rows)]))
    sel = rank < max_keys
    q_rows, pos, df, w = q_rows[sel], pos[sel], df[sel], w[sel]
    csum = np.cumsum(df)
    out = []
    lo = 0
    while lo < len(q_rows):
        base = csum[lo - 1] if lo else 0
        hi = int(np.searchsorted(csum, base + max_pairs, 'right'))
        hi = max(hi, lo + 1)
        if hi < len(q_rows):                       # do not split a query row across chunks
            hi2 = int(np.searchsorted(q_rows, q_rows[hi], 'left'))
            hi = hi2 if hi2 > lo else int(np.searchsorted(q_rows, q_rows[lo], 'right'))
        qr, ps, c, ww = q_rows[lo:hi], pos[lo:hi], df[lo:hi], w[lo:hi]
        tot = int(c.sum())
        offs = np.arange(tot) - np.repeat(np.cumsum(c) - c, c)
        pp = pr[np.repeat(first[ps], c) + offs]
        code = np.repeat(qr.astype(np.int64), c) * np.int64(n_pool) + pp
        wr = np.repeat(ww, c)
        uc, iv = np.unique(code, return_inverse=True)
        sc = np.bincount(iv, weights=wr).astype(np.float32)
        q = (uc // n_pool).astype(np.int32); p = (uc % n_pool).astype(np.int32)
        o = np.lexsort((-sc, q))
        q, p, sc = q[o], p[o], sc[o]
        st = np.r_[0, np.flatnonzero(np.diff(q)) + 1]
        rk = np.arange(len(q)) - np.repeat(st, np.diff(np.r_[st, len(q)]))
        m = rk < K
        out.append(pd.DataFrame({'q': q[m], 'p': p[m], 'score': sc[m], 'rank': rk[m].astype(np.int16)}))
        del offs, pp, code, wr, uc, iv; gc.collect()
        lo = hi
    return pd.concat(out, ignore_index=True) if out else empty

def generate_candidates(S, Pn, cfg):
    """S: query S1 rows, Pn: pool rows (same country). Returns candidate table q, p + blocking features."""
    res, keys = {}, {}
    for ch, K, cap, mk in (('name', cfg['k_name'], cfg['df_cap_name'], cfg['max_keys_name']),
                           ('addr', cfg['k_addr'], cfg['df_cap_addr'], cfg['max_keys_addr'])):
        keys[ch] = make_keys(Pn, ch), make_keys(S, ch)
        (pr, pk), (qr, qk) = keys[ch]
        res[ch] = block_channel(qr, qk, pr, pk, len(Pn), K, cap, mk, cfg['max_pairs_block'])
        del pr, pk, qr, qk; gc.collect()
    if cfg['k_joint'] > 0:
        # joint channel: name keys and address keys scored together, so a pool record sharing rare
        # name tokens AND rare address tokens outranks same-name branches elsewhere
        (pr1, pk1), (qr1, qk1) = keys['name']
        (pr2, pk2), (qr2, qk2) = keys['addr']
        res['joint'] = block_channel(np.concatenate([qr1, qr2]), np.concatenate([qk1, qk2]),
                                     np.concatenate([pr1, pr2]), np.concatenate([pk1, pk2]), len(Pn),
                                     cfg['k_joint'], cfg['df_cap_joint'], cfg['max_keys_joint'], cfg['max_pairs_block'])
        del pr1, pk1, qr1, qk1, pr2, pk2, qr2, qk2
    del keys; gc.collect()
    C = None
    for ch in ('name', 'addr', 'joint'):
        if ch not in res:
            continue
        d = res[ch].rename(columns={'score': f'{ch}_score', 'rank': f'{ch}_rank'})
        C = d if C is None else C.merge(d, on=['q', 'p'], how='outer')
    for ch in ('name', 'addr', 'joint'):
        if f'{ch}_score' not in C:
            C[f'{ch}_score'] = np.nan; C[f'{ch}_rank'] = np.nan
    C['n_channels'] = (C.name_score.notna().astype(np.int8) + C.addr_score.notna().astype(np.int8)
                       + C.joint_score.notna().astype(np.int8))
    C['cand_count'] = C.groupby('q').p.transform('size').astype(np.int16)
    for c in ('name_score', 'addr_score', 'joint_score', 'name_rank', 'addr_rank', 'joint_rank'):
        C[c] = C[c].astype(np.float32)
    return C.sort_values(['q', 'p']).reset_index(drop=True)

def countries_of(work_dir, split):
    c = load_norm(work_dir, split, 1, ['country']).country.value_counts()
    return [x for x in c.index if x != '']

# ============================================================ step 4
def step4_train_candidates(data_dir, work_dir, cfg=None):
    cfg = cfg_with(cfg)
    s1_all = load_norm(work_dir, 'train', 1, ['entity_id'])
    rs = np.random.RandomState(cfg['seed'])
    n = min(cfg['train_s1_sample'], len(s1_all))
    sample = set(s1_all.entity_id.values[rs.choice(len(s1_all), n, replace=False)])
    del s1_all
    _, gold = load_gold(data_dir)
    gold = gold[gold.s1.isin(sample)]
    parts, sample_rows = [], []
    for ctry in countries_of(work_dir, 'train'):
        S = load_norm(work_dir, 'train', 1, country=ctry)
        S = S[S.entity_id.isin(sample)].reset_index(drop=True)
        if not len(S):
            continue
        Pn = load_pool(work_dir, 'train', ctry)
        Pn = Pn.reset_index(drop=True)
        log(f'[{ctry}] blocking {len(S):,} train S1 against {len(Pn):,} pool records')
        C = generate_candidates(S, Pn, cfg)
        C['s1'] = S.entity_id.values[C.q.values]
        C['r'] = Pn.entity_id.values[C.p.values]
        C['country'] = ctry
        parts.append(C.drop(columns=['q', 'p']))
        sample_rows.append(S[['entity_id']].assign(country=ctry))
        log(f'[{ctry}] {len(C):,} candidates ({len(C) / len(S):.1f} per S1)')
        del S, Pn, C; gc.collect()
    C = pd.concat(parts, ignore_index=True)
    ids = pd.concat(sample_rows, ignore_index=True)
    g = gold.assign(y=np.int8(1))
    C = C.merge(g, on=['s1', 'r'], how='left')
    C['y'] = C.y.fillna(0).astype(np.int8)
    C.to_parquet(f'{work_dir}/train_cands.parquet', index=False)
    ids.to_parquet(f'{work_dir}/train_sample_s1.parquet', index=False)
    gold.to_parquet(f'{work_dir}/train_sample_gold.parquet', index=False)
    return recall_report(ids.entity_id.values, C, gold)

def recall_report(ids, C, gold):
    G = gold.groupby('s1').size().reindex(ids, fill_value=0)
    TPc = C[C.y == 1].groupby('s1').size().reindex(ids, fill_value=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ceil = np.where(G == 0, 1.0, 5 * TPc / (4 * TPc + G))
    rep = {'pair_recall': float(TPc.sum() / max(G.sum(), 1)),
           'entity_full_recall': float(((TPc == G) | (G == 0)).mean()),
           'ceiling_macro_f05': float(np.mean(ceil)),
           'candidates_per_s1': float(len(C) / len(ids)),
           'recall_name_channel': float(C[(C.y == 1) & C.name_score.notna()].shape[0] / max(G.sum(), 1)),
           'recall_addr_channel': float(C[(C.y == 1) & C.addr_score.notna()].shape[0] / max(G.sum(), 1)),
           'recall_joint_channel': float(C[(C.y == 1) & C.joint_score.notna()].shape[0] / max(G.sum(), 1))}
    if 'country' in C:
        for ctry, d in C.groupby('country'):
            s = d.s1.unique()
            rep[f'pair_recall_{ctry}'] = float(d.y.sum() / max(gold[gold.s1.isin(s)].shape[0], 1))
    for k, v in rep.items():
        log(f'  {k:28s} {v:.4f}')
    return rep

# ============================================================ features
FEATS = ['name_ratio', 'name_tsr', 'name_tsort', 'name_partial', 'core_tsr', 'core_jw', 'phon_tsr', 'phon_ratio',
         'core_exact', 'first_eq', 'legal_eq', 'legal_conflict', 'legal_a', 'legal_b',
         'addr_ratio', 'addr_tsr', 'addr_partial', 'addr_tsort', 'nums_tsr', 'house_eq', 'house_conflict',
         'house_absdiff', 'addr_empty_a', 'addr_empty_b', 'nums_empty_a', 'nums_empty_b',
         'name_ntok_a', 'name_ntok_b', 'addr_ntok_a', 'addr_ntok_b', 'tr_a', 'tr_b', 'is_s3',
         'name_score', 'name_rank', 'addr_score', 'addr_rank', 'joint_score', 'joint_rank', 'n_channels', 'cand_count',
         'name_x_addr']

def prep_side(df):
    """Adds precomputed per-record arrays used by pair features (in place, no copy: saves RAM)."""
    df['first_tok'] = df.name_core.str.split(n=1).str[0].fillna('')
    df['name_ntok'] = (df.name_n.str.count(' ') + (df.name_n != '')).astype(np.int16)
    df['addr_ntok'] = (df.addr_n.str.count(' ') + (df.addr_n != '')).astype(np.int16)
    df['house_num'] = pd.to_numeric(df.house.str.extract(r'^(\d{1,7})')[0], errors='coerce').astype(np.float32)
    df['tr'] = (df.name_tr | df.addr_tr).astype(np.int8)
    return df

def pair_features(C, A, B, workers=-1, chunk=1_500_000):
    """C has columns qi (row in A) and pi (row in B) plus blocking features. Returns float32 DataFrame FEATS."""
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import JaroWinkler
    out = []
    av = {c: A[c].values for c in A.columns}
    bv = {c: B[c].values for c in B.columns}
    for s in range(0, len(C), chunk):
        c = C.iloc[s:s + chunk]
        qi, pi = c.qi.values, c.pi.values
        a = lambda col: av[col][qi]
        b = lambda col: bv[col][pi]
        F = {}
        def sim(col, scorer, name, scale=1.0):
            x, y = a(col), b(col)
            v = process.cpdist(x, y, scorer=scorer, workers=workers).astype(np.float32) * scale
            v[(x == '') | (y == '')] = np.nan
            F[name] = v
        sim('name_n', fuzz.ratio, 'name_ratio'); sim('name_n', fuzz.token_set_ratio, 'name_tsr')
        sim('name_n', fuzz.token_sort_ratio, 'name_tsort'); sim('name_n', fuzz.partial_ratio, 'name_partial')
        sim('name_core', fuzz.token_set_ratio, 'core_tsr')
        sim('name_core', JaroWinkler.normalized_similarity, 'core_jw', 100.0)
        sim('name_phon', fuzz.token_set_ratio, 'phon_tsr'); sim('name_phon', fuzz.ratio, 'phon_ratio')
        sim('addr_n', fuzz.ratio, 'addr_ratio'); sim('addr_n', fuzz.token_set_ratio, 'addr_tsr')
        sim('addr_n', fuzz.partial_ratio, 'addr_partial'); sim('addr_n', fuzz.token_sort_ratio, 'addr_tsort')
        sim('nums', fuzz.token_set_ratio, 'nums_tsr')
        ca, cb = a('name_core'), b('name_core')
        F['core_exact'] = ((ca == cb) & (ca != '')).astype(np.float32)
        fa, fb = a('first_tok'), b('first_tok')
        F['first_eq'] = ((fa == fb) & (fa != '')).astype(np.float32)
        la, lb = a('legal'), b('legal')
        F['legal_a'] = (la != '').astype(np.float32); F['legal_b'] = (lb != '').astype(np.float32)
        F['legal_eq'] = ((la == lb) & (la != '')).astype(np.float32)
        F['legal_conflict'] = ((la != lb) & (la != '') & (lb != '')).astype(np.float32)
        ha, hb = a('house'), b('house')
        both = (ha != '') & (hb != '')
        F['house_eq'] = np.where(both, (ha == hb), np.nan).astype(np.float32)
        F['house_conflict'] = np.where(both, (ha != hb), np.nan).astype(np.float32)
        F['house_absdiff'] = np.abs(a('house_num') - b('house_num')).astype(np.float32)
        F['addr_empty_a'] = (a('addr_n') == '').astype(np.float32); F['addr_empty_b'] = (b('addr_n') == '').astype(np.float32)
        F['nums_empty_a'] = (a('nums') == '').astype(np.float32); F['nums_empty_b'] = (b('nums') == '').astype(np.float32)
        F['name_ntok_a'] = a('name_ntok').astype(np.float32); F['name_ntok_b'] = b('name_ntok').astype(np.float32)
        F['addr_ntok_a'] = a('addr_ntok').astype(np.float32); F['addr_ntok_b'] = b('addr_ntok').astype(np.float32)
        F['tr_a'] = a('tr').astype(np.float32); F['tr_b'] = b('tr').astype(np.float32)
        F['is_s3'] = b('is_s3').astype(np.float32)
        for col in ('name_score', 'name_rank', 'addr_score', 'addr_rank', 'joint_score', 'joint_rank', 'n_channels', 'cand_count'):
            F[col] = c[col].values.astype(np.float32)
        F['name_x_addr'] = (np.nan_to_num(F['name_tsr']) * np.nan_to_num(F['addr_tsr']) / 100).astype(np.float32)
        out.append(pd.DataFrame(F, index=c.index)[FEATS])
        del F; gc.collect()
    return pd.concat(out) if out else pd.DataFrame(columns=FEATS, dtype=np.float32)

def _index_side(df, ids):
    pos = pd.Series(np.arange(len(df)), index=df.entity_id.values)
    return pos.reindex(ids).values

# ============================================================ step 5
def step5_train_features(work_dir, cfg=None):
    cfg = cfg_with(cfg)
    C = pd.read_parquet(f'{work_dir}/train_cands.parquet')
    feats = []
    for ctry, c in C.groupby('country', sort=False):
        A = prep_side(load_norm(work_dir, 'train', 1, country=ctry).pipe(lambda d: d[d.entity_id.isin(set(c.s1))]).reset_index(drop=True))
        B = prep_side(load_pool(work_dir, 'train', ctry).pipe(lambda d: d[d.entity_id.isin(set(c.r))]).reset_index(drop=True))
        c = c.assign(qi=_index_side(A, c.s1.values), pi=_index_side(B, c.r.values))
        log(f'[{ctry}] features for {len(c):,} pairs')
        f = pair_features(c, A, B, chunk=cfg['feat_chunk'])
        feats.append(pd.concat([c[['s1', 'r', 'country', 'y']], f], axis=1))
        del A, B, c, f; gc.collect()
    T = pd.concat(feats).reset_index(drop=True)
    T.to_parquet(f'{work_dir}/train_feats.parquet', index=False)
    log(f'train feature table: {T.shape}')
    return T

# ============================================================ models
LGB_PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_bin=255, verbose=-1, seed=42)

def train_cv(X, y, groups, cfg, params=None, tag='2a'):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    p = dict(LGB_PARAMS, learning_rate=cfg['lgb_lr'], seed=cfg['seed'], num_threads=cfg['workers'])
    p.update(params or {})
    oof = np.zeros(len(y), np.float32)
    fold = np.zeros(len(y), np.int8)
    models = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=cfg['n_folds']).split(X, y, groups)):
        dtr = lgb.Dataset(X.iloc[tr], y[tr], free_raw_data=True)
        dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr)
        m = lgb.train(p, dtr, cfg['lgb_rounds'], valid_sets=[dva],
                      callbacks=[lgb.early_stopping(cfg['early_stop'], verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        fold[va] = k
        models.append(m)
        log(f'  stage {tag} fold {k}: best_iter={m.best_iteration}  logloss={m.best_score["valid_0"]["binary_logloss"]:.4f}')
    return oof, fold, models

def save_models(models, work_dir, tag):
    for k, m in enumerate(models):
        m.save_model(f'{work_dir}/model_{tag}_fold{k}.txt', num_iteration=m.best_iteration)

def load_models(work_dir, tag):
    import lightgbm as lgb
    paths = sorted(glob.glob(f'{work_dir}/model_{tag}_fold*.txt'))
    assert paths, f'no saved models for stage {tag}'
    return [lgb.Booster(model_file=p) for p in paths]

def predict_avg(models, X):
    return np.mean([m.predict(X) for m in models], axis=0).astype(np.float32)

def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float('nan')

# ============================================================ step 6
def step6_train_stage2a(work_dir, cfg=None, run_loco=True):
    cfg = cfg_with(cfg)
    T = pd.read_parquet(f'{work_dir}/train_feats.parquet')
    X, y = T[FEATS], T.y.values
    log(f'stage 2a: {len(T):,} pairs, {int(y.sum()):,} positives, {T.s1.nunique():,} S1')
    oof, fold, models = train_cv(X, y, T.s1.values, cfg, tag='2a')
    save_models(models, work_dir, '2a')
    T[['s1', 'r', 'country', 'y']].assign(p2a=oof, fold=fold).to_parquet(f'{work_dir}/oof_2a.parquet', index=False)
    rep = {'oof_auc_2a': auc(y, oof)}
    imp = pd.Series(np.mean([m.feature_importance('gain') for m in models], axis=0), index=FEATS)
    rep['top_features'] = imp.sort_values(ascending=False).head(15).round(0).to_dict()
    if run_loco and T.country.nunique() > 1:
        import lightgbm as lgb
        p = dict(LGB_PARAMS, learning_rate=cfg['lgb_lr'], num_threads=cfg['workers'])
        n_it = int(np.mean([m.best_iteration for m in models])) or 200
        for ctry in T.country.unique():
            tr, va = T.country.values != ctry, T.country.values == ctry
            m = lgb.train(p, lgb.Dataset(X[tr], y[tr]), n_it)
            rep[f'loco_auc_{ctry}'] = auc(y[va], m.predict(X[va]))
            rep[f'in_country_auc_{ctry}'] = auc(y[va], oof[va])
    for k, v in rep.items():
        log(f'  {k}: {v}')
    return rep

# ============================================================ group context (stage 2b)
BASE_2B = ['name_tsr', 'addr_tsr', 'core_tsr', 'house_conflict', 'name_rank', 'addr_rank', 'joint_rank', 'n_channels', 'is_s3']
G_FEATS = ['logit2a', 'rank2a', 'gap_best', 'p_best', 'p_second', 'n_above50', 'sum_p', 'rel_best',
           'cand_count'] + BASE_2B

def group_features(D):
    """D: s1, p2a, cand_count + BASE_2B columns. Adds within-S1 context features."""
    D = D.copy()
    p = D.p2a.values.astype(np.float64).clip(1e-6, 1 - 1e-6)
    D['logit2a'] = np.log(p / (1 - p)).astype(np.float32)
    g = D.groupby('s1', sort=False).p2a
    D['rank2a'] = g.rank(ascending=False, method='first').astype(np.float32)
    D['p_best'] = g.transform('max').astype(np.float32)
    D['gap_best'] = (D.p_best - D.p2a).astype(np.float32)
    D['sum_p'] = g.transform('sum').astype(np.float32)
    D['n_above50'] = D.p2a.gt(0.5).groupby(D.s1, sort=False).transform('sum').astype(np.float32)
    second = D.p2a.where(D.rank2a == 2).groupby(D.s1, sort=False).transform('max').fillna(0)
    D['p_second'] = second.astype(np.float32)
    D['rel_best'] = (D.p2a / D.p_best.clip(lower=1e-6)).astype(np.float32)
    return D

def step7_train_stage2b(work_dir, cfg=None):
    cfg = cfg_with(cfg)
    T = pd.read_parquet(f'{work_dir}/train_feats.parquet', columns=['s1', 'r', 'y'] + BASE_2B + ['cand_count'])
    O = pd.read_parquet(f'{work_dir}/oof_2a.parquet', columns=['p2a', 'fold'])
    D = group_features(pd.concat([T, O], axis=1))
    oof, fold, models = train_cv(D[G_FEATS], D.y.values, D.s1.values, cfg,
                                 params=dict(num_leaves=31, min_data_in_leaf=200), tag='2b')
    save_models(models, work_dir, '2b')
    D[['s1', 'r', 'y', 'p2a']].assign(p2b=oof, fold=fold).to_parquet(f'{work_dir}/oof_2b.parquet', index=False)
    rep = {'oof_auc_2a': auc(D.y.values, D.p2a.values), 'oof_auc_2b': auc(D.y.values, oof)}
    for k, v in rep.items():
        log(f'  {k}: {v:.5f}')
    return rep

# ============================================================ calibration + decision (step 8)
def to_matrix(ids, s1, p, y=None, K=20):
    """Pads each S1's candidates (sorted by p desc) into an (n_ids, K) matrix."""
    idx = pd.Index(ids)
    qi = idx.get_indexer(s1)
    o = np.lexsort((-p, qi))
    qi, p = qi[o], p[o]
    st = np.r_[0, np.flatnonzero(np.diff(qi)) + 1] if len(qi) else np.zeros(0, int)
    rk = np.arange(len(qi)) - np.repeat(st, np.diff(np.r_[st, len(qi)]))
    m = rk < K
    M = np.zeros((len(idx), K), np.float32)
    M[qi[m], rk[m]] = p[m]
    Y = None
    if y is not None:
        Y = np.zeros((len(idx), K), np.int8)
        Y[qi[m], rk[m]] = np.asarray(y)[o][m]
    return M, Y, (qi, rk, o)

def decide_k(M, rule, params):
    K = M.shape[1]
    if rule == 'global':
        return (M >= params['t']).sum(1)
    if rule == 'two_thr':
        first = M[:, 0] >= params['t1']
        return np.where(first, 1 + (M[:, 1:] >= params['t2']).sum(1), 0)
    if rule == 'expected_f':
        P = M.astype(np.float64)
        if params.get('beta', 0):
            with np.errstate(divide='ignore'):
                lg = np.log(np.clip(P, 1e-9, 1 - 1e-9) / (1 - np.clip(P, 1e-9, 1 - 1e-9))) + params['beta']
            P = np.where(M > 0, 1 / (1 + np.exp(-lg)), 0)
        S = np.cumsum(P, 1)
        Gexp = S[:, -1:] + params.get('lam', 0.0)
        kk = np.arange(1, K + 1)[None, :]
        val = 5 * S / (4 * kk + Gexp)
        v0 = np.prod(1 - P, 1)
        best = val.argmax(1)
        return np.where(v0 >= val[np.arange(len(M)), best], 0, best + 1)
    raise ValueError(rule)

def eval_k(k, Y, G):
    cs = np.concatenate([np.zeros((len(Y), 1), np.int32), np.cumsum(Y, 1)], 1)
    tp = cs[np.arange(len(Y)), k]
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.where(G == 0, (k == 0).astype(float), 5 * tp / (4 * k + G))
    return f

def tune_rules(M, Y, G):
    res = {}
    best = ('global', {'t': 0.5}, -1)
    for t in np.round(np.linspace(0.05, 0.95, 91), 3):
        f = eval_k(decide_k(M, 'global', {'t': t}), Y, G).mean()
        if f > res.get('global', (0, -1))[1]:
            res['global'] = ({'t': float(t)}, f)
    for t1 in np.round(np.linspace(0.1, 0.9, 17), 3):
        for t2 in np.round(np.linspace(0.1, 0.95, 18), 3):
            f = eval_k(decide_k(M, 'two_thr', {'t1': t1, 't2': t2}), Y, G).mean()
            if f > res.get('two_thr', (0, -1))[1]:
                res['two_thr'] = ({'t1': float(t1), 't2': float(t2)}, f)
    for beta in np.round(np.linspace(-2, 2, 41), 2):
        f = eval_k(decide_k(M, 'expected_f', {'beta': beta}), Y, G).mean()
        if f > res.get('expected_f', (0, -1))[1]:
            res['expected_f'] = ({'beta': float(beta)}, f)
    for rule, (prm, f) in res.items():
        if f > best[2]:
            best = (rule, prm, f)
    return best, {r: (prm, round(float(f), 5)) for r, (prm, f) in res.items()}

def step8_decide(work_dir, cfg=None):
    from sklearn.isotonic import IsotonicRegression
    cfg = cfg_with(cfg)
    O = pd.read_parquet(f'{work_dir}/oof_2b.parquet')
    ids = pd.read_parquet(f'{work_dir}/train_sample_s1.parquet').entity_id.values
    gold = pd.read_parquet(f'{work_dir}/train_sample_gold.parquet')
    G = gold.groupby('s1').size().reindex(ids, fill_value=0).values
    # cross-fitted isotonic calibration (each fold calibrated by the other folds)
    pc = np.zeros(len(O), np.float32)
    for k in np.unique(O.fold):
        tr, va = O.fold.values != k, O.fold.values == k
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(O.p2b.values[tr], O.y.values[tr])
        pc[va] = iso.predict(O.p2b.values[va])
    iso_all = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(O.p2b.values, O.y.values)
    import pickle
    with open(f'{work_dir}/isotonic.pkl', 'wb') as f:
        pickle.dump(iso_all, f)
    report = {}
    for name, p in (('stage2a_raw', O.p2a.values), ('stage2b_raw', O.p2b.values), ('stage2b_calibrated', pc)):
        M, Y, _ = to_matrix(ids, O.s1.values, p, O.y.values, cfg['kmax'])
        best, allr = tune_rules(M, Y, G)
        report[name] = {'best_rule': best[0], 'params': best[1], 'cv_macro_f05': round(float(best[2]), 5), 'all_rules': allr}
        log(f'  {name:20s} best={best[0]} {best[1]}  CV macro F0.5 = {best[2]:.4f}')
    # final choice: best of the calibrated stage-2b probabilities (what test inference will use)
    fin = report['stage2b_calibrated']
    decision = {'rule': fin['best_rule'], 'params': fin['params'], 'cv_macro_f05': fin['cv_macro_f05']}
    M, Y, _ = to_matrix(ids, O.s1.values, pc, O.y.values, cfg['kmax'])
    k = decide_k(M, decision['rule'], decision['params'])
    f = eval_k(k, Y, G)
    decision['singleton_accuracy'] = float((k[G == 0] == 0).mean()) if (G == 0).any() else None
    decision['non_singleton_abstention'] = float((k[G > 0] == 0).mean())
    json.dump(decision, open(f'{work_dir}/decision.json', 'w'), indent=1)
    O.assign(pcal=pc).to_parquet(f'{work_dir}/oof_final.parquet', index=False)
    report['final'] = decision
    log(f'FINAL CV macro F0.5 = {decision["cv_macro_f05"]:.4f}   decision: {decision["rule"]} {decision["params"]}')
    return report

# ============================================================ step 9
def step9_error_analysis(work_dir, cfg=None, n_examples=8):
    cfg = cfg_with(cfg)
    O = pd.read_parquet(f'{work_dir}/oof_final.parquet')
    dec = json.load(open(f'{work_dir}/decision.json'))
    ids = pd.read_parquet(f'{work_dir}/train_sample_s1.parquet')
    gold = pd.read_parquet(f'{work_dir}/train_sample_gold.parquet')
    G = gold.groupby('s1').size().reindex(ids.entity_id, fill_value=0).values
    M, Y, (qi, rk, o) = to_matrix(ids.entity_id.values, O.s1.values, O.pcal.values, O.y.values, cfg['kmax'])
    k = decide_k(M, dec['rule'], dec['params'])
    f = eval_k(k, Y, G)
    Os = O.iloc[o].reset_index(drop=True)
    Os['rk'] = rk
    Os['pred'] = rk < k[qi]
    T = pd.read_parquet(f'{work_dir}/train_feats.parquet', columns=['s1', 'r', 'country', 'name_tsr', 'addr_tsr', 'house_conflict', 'tr_a', 'tr_b', 'core_exact'])
    Os = Os.merge(T, on=['s1', 'r'], how='left')
    in_c = O[O.y == 1].groupby('s1').size().reindex(ids.entity_id, fill_value=0).values
    rep = {
        'cv_macro_f05': float(f.mean()),
        'S1_with_blocking_miss': float((in_c < G).mean()),
        'gold_pairs_missed_by_blocking': int(G.sum() - in_c.sum()),
        'false_positive_pairs': int(((Os.pred) & (Os.y == 0)).sum()),
        'false_negative_pairs_in_candidates': int(((~Os.pred) & (Os.y == 1)).sum()),
        'true_positive_pairs': int(((Os.pred) & (Os.y == 1)).sum()),
        'singleton_false_positive_S1': int(((G == 0) & (k > 0)).sum()),
    }
    fp_ = Os[(Os.pred) & (Os.y == 0)]
    fn_ = Os[(~Os.pred) & (Os.y == 1)]
    rep['FP_same_core_name'] = float(fp_.core_exact.mean()) if len(fp_) else None
    rep['FP_house_conflict'] = float((fp_.house_conflict == 1).mean()) if len(fp_) else None
    rep['FN_cross_script'] = float(((fn_.tr_a + fn_.tr_b) > 0).mean()) if len(fn_) else None
    rep['FN_low_name_sim'] = float((fn_.name_tsr < 50).mean()) if len(fn_) else None
    per_c = pd.Series(f, index=ids.entity_id.values).groupby(ids.country.values).mean()
    rep['cv_f05_by_country'] = per_c.round(4).to_dict()
    for kk, v in rep.items():
        log(f'  {kk:36s} {v}')
    # examples with raw text
    def show(df, title):
        if not len(df):
            return
        print(f'\n--- {title} ---')
        print(df[['s1', 'r', 'pcal', 'name_tsr', 'addr_tsr']].head(n_examples).to_string(index=False))
    show(fp_.sort_values('pcal', ascending=False), 'most confident FALSE POSITIVES')
    show(fn_.sort_values('pcal'), 'least confident FALSE NEGATIVES')
    return rep

# ============================================================ submission writing / validation
def write_submission(out_dir, s1_ids, cand, match):
    """cand/match: DataFrames s1, r. Writes both TSVs with one row per test S1 in file order."""
    os.makedirs(out_dir, exist_ok=True)
    for fname, col, df in (('candidate_pairs.tsv', 'candidate_entity_ids', cand),
                           ('matching_results.tsv', 'matched_entity_ids', match)):
        d = df.drop_duplicates(['s1', 'r'])
        lists = d.groupby('s1', sort=False).r.agg(','.join)
        s = pd.Series(lists.reindex(s1_ids).fillna('').values, index=s1_ids)
        out = pd.DataFrame({'source1_entity_id': s1_ids, col: s.values})
        out.to_csv(f'{out_dir}/{fname}', sep='\t', index=False, quoting=csv.QUOTE_NONE, lineterminator='\n')
        log(f'wrote {out_dir}/{fname}: {len(out):,} rows, {len(d):,} ids')

def validate_outputs(out_dir, work_dir, data_dir):
    rep = {}
    s1 = load_norm(work_dir, 'test', 1, ['entity_id']).entity_id.values
    pool = np.concatenate([fp(load_norm(work_dir, 'test', k, ['entity_id']).entity_id.values) for k in (2, 3)])
    lists = {}
    for fname, col in (('matching_results.tsv', 'matched_entity_ids'), ('candidate_pairs.tsv', 'candidate_entity_ids')):
        d = read_tsv(f'{out_dir}/{fname}', ['source1_entity_id', col])
        e = d.assign(r=d[col].str.split(',')).explode('r')
        e = e[e.r.notna() & (e.r != '')]
        rep[fname] = {
            'rows': len(d), 'rows_equal_test_s1': bool(len(d) == len(s1) and set(d.source1_entity_id) == set(s1)),
            'duplicate_s1_rows': int(d.source1_entity_id.duplicated().sum()),
            'bad_prefix': int((~e.r.str[:3].isin(['S2-', 'S3-'])).sum()),
            'ids_not_in_test_pool': int(np.isin(fp(e.r.values), pool, invert=True).sum()),
            'duplicates_in_list': int(e.duplicated(['source1_entity_id', 'r']).sum()),
            'nan_strings': int(d[col].isin(['nan', 'None', '[]']).sum()),
        }
        lists[fname] = e[['source1_entity_id', 'r']]
    m = lists['matching_results.tsv'].merge(lists['candidate_pairs.tsv'], how='left', indicator=True)
    rep['matches_not_in_candidates'] = int((m._merge == 'left_only').sum())
    ok = all(v['rows_equal_test_s1'] and v['duplicate_s1_rows'] == 0 and v['bad_prefix'] == 0 and
             v['ids_not_in_test_pool'] == 0 and v['duplicates_in_list'] == 0 and v['nan_strings'] == 0
             for k, v in rep.items() if isinstance(v, dict)) and rep['matches_not_in_candidates'] == 0
    rep['ALL_CHECKS_PASS'] = bool(ok)
    # official validator, if present anywhere near the data
    dd = os.path.abspath(data_dir)
    cands = [p for pat in (f'{dd}/validate_submission.py', f'{dd}/*/validate_submission.py',
                           f'{os.path.dirname(dd)}/validate_submission.py', f'{os.path.dirname(dd)}/*/validate_submission.py',
                           os.path.join(os.getcwd(), 'utils', 'validate_submission.py'))
             for p in glob.glob(pat)]
    if cands:
        r = subprocess.run([sys.executable, cands[0], '--matching', f'{out_dir}/matching_results.tsv',
                            '--candidate', f'{out_dir}/candidate_pairs.tsv', '--test-dir', data_dir],
                           capture_output=True, text=True)
        rep['official_validator'] = (r.stdout + r.stderr)[-1500:]
    else:
        rep['official_validator'] = 'validate_submission.py not found (upload utils/ next to the data to run it)'
    log(f'validation: {"PASS" if ok else "FAIL"}  {rep}')
    return rep

# ============================================================ step 11: test inference
def step11_predict_test(data_dir, work_dir, out_dir, cfg=None, force=False):
    import pickle
    cfg = cfg_with(cfg)
    m2a, m2b = load_models(work_dir, '2a'), load_models(work_dir, '2b')
    iso = pickle.load(open(f'{work_dir}/isotonic.pkl', 'rb'))
    dec = json.load(open(f'{work_dir}/decision.json'))
    for ctry in countries_of(work_dir, 'test'):
        path = f'{work_dir}/test_scored_{re.sub("[^a-z0-9]", "_", ctry)}.parquet'
        if os.path.exists(path) and not force:
            log(f'[{ctry}] already scored, skipped'); continue
        S = load_norm(work_dir, 'test', 1, country=ctry).reset_index(drop=True)
        Pn = load_pool(work_dir, 'test', ctry).reset_index(drop=True)
        log(f'[{ctry}] test: {len(S):,} S1 vs {len(Pn):,} pool records')
        C = generate_candidates(S, Pn, cfg)
        log(f'[{ctry}] {len(C):,} candidates ({len(C) / max(len(S), 1):.1f} per S1)')
        A, B = prep_side(S), prep_side(Pn)
        C = C.rename(columns={'q': 'qi', 'p': 'pi'})
        keep = []
        for s in range(0, len(C), cfg['feat_chunk']):
            c = C.iloc[s:s + cfg['feat_chunk']]
            F = pair_features(c, A, B, chunk=cfg['feat_chunk'])
            p2a = predict_avg(m2a, F[FEATS])
            keep.append(pd.DataFrame({'qi': c.qi.values, 'pi': c.pi.values, 'p2a': p2a,
                                      **{b: F[b].values for b in BASE_2B + ['cand_count']}}))
            log(f'[{ctry}] scored {min(s + cfg["feat_chunk"], len(C)):,}/{len(C):,}')
            del F; gc.collect()
        D = pd.concat(keep, ignore_index=True)
        D['s1'] = S.entity_id.values[D.qi.values]
        D['r'] = Pn.entity_id.values[D.pi.values]
        D = group_features(D)
        D['p2b'] = predict_avg(m2b, D[G_FEATS])
        D['pcal'] = iso.predict(D.p2b.values).astype(np.float32)
        D[['s1', 'r', 'p2a', 'p2b', 'pcal']].to_parquet(path, index=False)
        del S, Pn, A, B, C, D, keep; gc.collect()
    # decision + exclusivity over all countries
    s1_ids = load_norm(work_dir, 'test', 1, ['entity_id']).entity_id.values
    D = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{work_dir}/test_scored_*.parquet'))], ignore_index=True)
    M, _, (qi, rk, o) = to_matrix(s1_ids, D.s1.values, D.pcal.values, None, cfg['kmax'])
    k = decide_k(M, dec['rule'], dec['params'])
    Ds = D.iloc[o].reset_index(drop=True)
    sel = Ds[rk < k[qi]]
    sel = sel.sort_values('pcal', ascending=False).drop_duplicates('r')     # exclusivity (gate G1)
    log(f'test: {len(D):,} candidate pairs, {len(sel):,} predicted matches, '
        f'{(k == 0).mean():.1%} S1 predicted empty, mean |pred| = {len(sel) / len(s1_ids):.2f}')
    write_submission(out_dir, s1_ids, D[['s1', 'r']], sel[['s1', 'r']])
    return validate_outputs(out_dir, work_dir, data_dir)

# ============================================================ packaging
README = """# Business Entity Resolution - CPFC pipeline

Calibrated Precision-First Cascade: normalization -> multi-channel blocking -> LightGBM pair model
-> group-context LightGBM -> isotonic calibration -> per-entity F0.5 decision + exclusivity.

## Reproduce
    pip install -r requirements.txt
    python src/run_pipeline.py --data-dir <folder with the 7 TSV files>

Outputs: output/matching_results.tsv and output/candidate_pairs.tsv (candidate_pairs = exact set scored by the model).
No external data, no external APIs, no pretrained models; libraries: pandas, numpy, pyarrow, rapidfuzz (MIT),
lightgbm (MIT), scikit-learn (BSD).
"""
RUNNER = """import argparse, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ber_lib as B

ap = argparse.ArgumentParser()
ap.add_argument('--data-dir', required=True); ap.add_argument('--work-dir', default='work')
ap.add_argument('--out-dir', default='output'); ap.add_argument('--cfg', default='{}')
a = ap.parse_args()
cfg = B.cfg_with(json.loads(a.cfg))
B.step2_normalize(a.data_dir, a.work_dir, cfg)
B.step4_train_candidates(a.data_dir, a.work_dir, cfg)
B.step5_train_features(a.work_dir, cfg)
B.step6_train_stage2a(a.work_dir, cfg)
B.step7_train_stage2b(a.work_dir, cfg)
B.step8_decide(a.work_dir, cfg)
B.step11_predict_test(a.data_dir, a.work_dir, a.out_dir, cfg)
"""
DOC = """# Documentation - Business Entity Resolution

## Methodology
Calibrated Precision-First Cascade (CPFC). Source 1 is the deduplicated reference; for every S1 entity we retrieve
candidates from Source 2/3 of the same country, score every pair, calibrate, and choose the per-entity match set that
maximizes F0.5. Each S2/S3 record is assigned to at most one S1 (exclusivity, verified on the training ground truth).

## Normalization
NFKC, case folding, Indic-script transliteration built from Unicode character names (no external data), accent
folding, punctuation removal, joining of spelled-out initials (L.L.C. -> llc), legal-form extraction, state/street
abbreviation contraction, house number and numeric-token extraction, consonant-skeleton phonetic keys.

## Blocking
Three channels, each an inverted index over rare keys (document frequency <= cap), ranked by summed IDF:
name channel (core-name tokens, bigrams, phonetic keys, exact core name), address channel (address tokens and
bigrams) and a joint channel scoring name and address keys together (separates same-name branches).
Top-{k_name} + top-{k_addr} + top-{k_joint} per S1, union. candidate_pairs.tsv is exactly this union.

## Model
Stage 2a: LightGBM on {n_feats} pair features (string similarities at several normalization levels, phonetic
similarity, legal-form agreement, house-number agreement/conflict, numeric-token overlap, missingness, blocking scores).
5-fold GroupKFold by S1. Stage 2b: LightGBM on out-of-fold stage-2a probabilities with within-entity context
(rank, gap to best, competition). Isotonic calibration (cross-fitted).

## Decision
Tuned on out-of-fold predictions: {rule} {params}. Cross-validated macro F0.5 on the training sample: {cv}.

## Results
{results}
"""

def package_submission(work_dir, out_dir, lib_path, zip_path, results=None, cfg=None):
    cfg = cfg_with(cfg)
    stage = os.path.join(os.path.dirname(zip_path), '_pkg')
    shutil.rmtree(stage, ignore_errors=True)
    src = f'{stage}/code/business_entity_resolution/src'
    os.makedirs(src); os.makedirs(f'{stage}/output')
    shutil.copy(lib_path, f'{src}/ber_lib.py')
    runner = os.path.join(os.path.dirname(os.path.abspath(lib_path)), 'run_pipeline.py')
    if os.path.exists(runner):
        shutil.copy(runner, f'{src}/run_pipeline.py')
    else:
        open(f'{src}/run_pipeline.py', 'w').write(RUNNER)
    open(f'{stage}/code/business_entity_resolution/README.md', 'w').write(README)
    open(f'{stage}/code/business_entity_resolution/requirements.txt', 'w').write(
        'pandas>=2.0\nnumpy>=1.24\npyarrow>=12\nrapidfuzz>=3.6\nlightgbm>=4.0\nscikit-learn>=1.3\npsutil\n')
    dec = json.load(open(f'{work_dir}/decision.json'))
    open(f'{stage}/Documentation_template.md', 'w').write(DOC.format(
        k_name=cfg['k_name'], k_addr=cfg['k_addr'], k_joint=cfg['k_joint'], n_feats=len(FEATS), rule=dec['rule'], params=dec['params'],
        cv=dec['cv_macro_f05'], results=json.dumps(results or {}, indent=1, default=str)))
    for f in ('matching_results.tsv', 'candidate_pairs.tsv'):
        shutil.copy(f'{out_dir}/{f}', f'{stage}/output/{f}')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(stage):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, stage))
    shutil.rmtree(stage, ignore_errors=True)
    log(f'package written: {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)')
    return zip_path
