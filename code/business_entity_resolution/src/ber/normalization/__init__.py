"""Normalization and structured representation pipeline."""
from ber.normalization.unicode import normalize_unicode, strip_accents
from ber.normalization.fold import fold_token, fold_text
from ber.normalization.name import normalize_business_name
from ber.normalization.address import normalize_address
from ber.normalization.country import normalize_country
from ber.normalization.mining import (
    is_abbrev,
    compute_abbrev_align,
    mine_abbreviations_from_pairs,
)

from ber.normalization.pipeline import canonicalize_record

__all__ = [
    "normalize_unicode",
    "strip_accents",
    "fold_token",
    "fold_text",
    "normalize_business_name",
    "normalize_address",
    "normalize_country",
    "is_abbrev",
    "compute_abbrev_align",
    "mine_abbreviations_from_pairs",
    "canonicalize_record",
]
