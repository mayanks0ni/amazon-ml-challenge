"""Structured information extraction from business names and addresses."""
from ber.extraction.legal_form import (
    extract_legal_form,
    check_legal_form_compatibility,
)
from ber.extraction.dba import extract_dba_aliases
from ber.extraction.numbers import (
    extract_address_numbers,
    compare_house_numbers,
    compare_unit_numbers,
)
from ber.extraction.postal import extract_postal_code, compare_postal_codes
from ber.extraction.landmark import extract_landmark_phrase

__all__ = [
    "extract_legal_form",
    "check_legal_form_compatibility",
    "extract_dba_aliases",
    "extract_address_numbers",
    "compare_house_numbers",
    "compare_unit_numbers",
    "extract_postal_code",
    "compare_postal_codes",
    "extract_landmark_phrase",
]
