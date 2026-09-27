"""Indic script transliteration to Latin phonetic characters.

Maps Unicode Indic blocks (Devanagari, Bengali, Gurmukhi, Gujarati,
Oriya, Tamil, Telugu, Kannada, Malayalam) to ASCII phonetic equivalents.
Enables cross-lingual entity resolution without external GPL libraries.
"""
from typing import Dict, List, Tuple

# Mapping of relative offset within any 128-byte Unicode Indic block
INDIC_OFFSET_MAP: Dict[int, str] = {
    0x01: "n",   # Chandrabindu
    0x02: "n",   # Anusvara (ं)
    0x03: "h",   # Visarga (ः)
    0x05: "a",   # अ
    0x06: "a",   # आ
    0x07: "i",   # इ
    0x08: "i",   # ई
    0x09: "u",   # उ
    0x0A: "u",   # ऊ
    0x0B: "r",   # ऋ
    0x0E: "e",   # ऎ
    0x0F: "e",   # ए
    0x10: "ai",  # ऐ
    0x11: "o",   # ऒ
    0x12: "o",   # ओ
    0x13: "au",  # औ
    0x14: "au",
    0x15: "k",   # क
    0x16: "kh",  # ख
    0x17: "g",   # ग
    0x18: "gh",  # घ
    0x19: "ng",  # ङ
    0x1A: "c",   # च
    0x1B: "ch",  # छ
    0x1C: "j",   # ज
    0x1D: "jh",  # झ
    0x1E: "ny",  # ञ
    0x1F: "t",   # ट
    0x20: "th",  # ठ
    0x21: "d",   # ड
    0x22: "dh",  # ढ
    0x23: "n",   # ण
    0x24: "t",   # त
    0x25: "th",  # थ
    0x26: "d",   # द
    0x27: "dh",  # ध
    0x28: "n",   # न
    0x2A: "p",   # प
    0x2B: "ph",  # फ
    0x2C: "b",   # ब
    0x2D: "bh",  # भ
    0x2E: "m",   # म
    0x2F: "y",   # य
    0x30: "r",   # र
    0x31: "r",   # ऱ
    0x32: "l",   # ल
    0x33: "l",   # ळ
    0x34: "zh",  # ழ
    0x35: "v",   # व
    0x36: "sh",  # श
    0x37: "sh",  # ष
    0x38: "s",   # स
    0x39: "h",   # ह
    0x3C: "",    # Nukta
    0x3E: "a",   # ा (aa matra)
    0x3F: "i",   # ि (i matra)
    0x40: "i",   # ी (ii matra)
    0x41: "u",   # ु (u matra)
    0x42: "u",   # ू (uu matra)
    0x43: "r",   # ृ
    0x46: "e",   # ॆ
    0x47: "e",   # े
    0x48: "ai",  # ै
    0x4A: "o",   # ॊ
    0x4B: "o",   # ो
    0x4C: "au",  # ौ
    0x4D: "",    # Virama / Halant (suppresses default vowel)
    0x4E: "e",   # ॅ
    0x4F: "o",   # ॉ
    0x58: "q",   # क़
    0x59: "kh",  # ख़
    0x5A: "g",   # ग़
    0x5B: "z",   # ज़
    0x5C: "r",   # ड़
    0x5D: "rh",  # ढ़
    0x5E: "f",   # फ़
    0x5F: "y",   # य़
    0x66: "0", 0x67: "1", 0x68: "2", 0x69: "3", 0x6A: "4",  # Indic digits
    0x6B: "5", 0x6C: "6", 0x6D: "7", 0x6E: "8", 0x6F: "9",
}

INDIC_BLOCK_RANGES: List[Tuple[int, int]] = [
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Oriya
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
]


def has_indic_characters(text: str) -> bool:
    """Returns True if the text contains any Brahmic/Indic script characters."""
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x0D7F:
            return True
    return False


def transliterate_indic(text: str) -> str:
    """Converts Indic script characters in text to Latin phonetic characters."""
    if not text or not has_indic_characters(text):
        return text

    chars: List[str] = []
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x0D7F:
            matched = False
            for start, end in INDIC_BLOCK_RANGES:
                if start <= cp <= end:
                    offset = cp - start
                    chars.append(INDIC_OFFSET_MAP.get(offset, ""))
                    matched = True
                    break
            if not matched:
                chars.append(ch)
        else:
            chars.append(ch)

    return "".join(chars)
