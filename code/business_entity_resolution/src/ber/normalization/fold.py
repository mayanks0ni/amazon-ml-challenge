"""Symmetric transliteration folding rules from Table 10.3."""
import re
from ber.normalization.unicode import normalize_unicode, strip_accents

# Table 10.3 replacement patterns
VOWEL_REPLACEMENTS = [
    (r"aa+", "a"),
    (r"ee+", "i"),
    (r"ii+", "i"),
    (r"oo+", "u"),
    (r"ou+", "u"),
]

ASPIRATE_REPLACEMENTS = [
    (r"bh", "b"),
    (r"dh", "d"),
    (r"th", "t"),
    (r"kh", "k"),
    (r"gh", "g"),
    (r"ph", "f"),
    (r"jh", "j"),
    (r"ch", "c"),
]

CONSONANT_REPLACEMENTS = [
    (r"sh", "s"),
    (r"w", "v"),
    (r"z", "j"),
    (r"ck", "k"),
    (r"q", "k"),
]

DOUBLE_LETTER_RE = re.compile(r"([a-z])\1+")


def fold_token(token: str) -> str:
    """Folds a single word token according to symmetric transliteration rules."""
    if len(token) <= 1:
        return token

    t = token
    # 1. Vowel variants (shree/shri -> sri, raaj/raj -> raj, noor/nur -> nur)
    for pat, repl in VOWEL_REPLACEMENTS:
        t = re.sub(pat, repl, t)

    # 2. Collapse doubled letters (agarwal/aggarwal -> agarwal, galli/gali -> gali)
    t = re.sub(r"([a-z])\1+", r"\1", t)

    # 3. Aspirate folding (bharat/barat -> barat, sidharth/sidarth -> sidart)
    for pat, repl in ASPIRATE_REPLACEMENTS:
        t = re.sub(pat, repl, t)

    # 4. Consonant variants (shiv/siv -> siv, vishnu/wishnu -> visnu)
    for pat, repl in CONSONANT_REPLACEMENTS:
        t = re.sub(pat, repl, t)

    # 5. Schwa deletion: trailing 'a' after consonant dropped for tokens of length >= 4
    # (ganesha/ganesh -> ganesh, krishna/krishn -> krisn)
    if len(t) >= 4 and t.endswith("a") and t[-2] not in "aeiou":
        t = t[:-1]

    # 6. Final vowel spelling: y -> i at word end (shanty/shanti -> santi)
    if t.endswith("y"):
        t = t[:-1] + "i"

    # Re-collapse any doubles that may have resulted from reductions
    t = DOUBLE_LETTER_RE.sub(r"\1", t)
    return t


def fold_text(text: str) -> str:
    """Applies transliteration folding across all word tokens in the text."""
    clean = strip_accents(normalize_unicode(text))
    tokens = re.split(r"[^a-z0-9]+", clean)
    folded_tokens = [fold_token(tok) for tok in tokens if tok]
    return " ".join(folded_tokens)
