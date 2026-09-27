"""Unit tests for normalization and extraction components."""
import pytest
from ber.normalization.unicode import normalize_unicode, strip_accents
from ber.normalization.fold import fold_token, fold_text
from ber.extraction.legal_form import extract_legal_form, check_legal_form_compatibility
from ber.extraction.dba import extract_dba_aliases
from ber.extraction.numbers import extract_address_numbers, compare_house_numbers
from ber.extraction.postal import extract_postal_code, compare_postal_codes
from ber.extraction.landmark import extract_landmark_phrase
from ber.normalization.mining import is_abbrev, compute_abbrev_align
from ber.io.schemas import RawRecord
from ber.normalization.pipeline import canonicalize_record


def test_unicode_and_accents():
    # Café -> cafe, Société -> societe
    assert normalize_unicode("Café") == "café"
    assert strip_accents("Café") == "Cafe"
    assert strip_accents("Société") == "Societe"


def test_transliteration_folding():
    # Table 10.3 rules
    # Double letter collapse
    assert fold_token("aggarwal") == fold_token("agarwal")
    assert fold_token("galli") == fold_token("gali")
    # Vowels
    assert fold_token("shree") == fold_token("shri")
    assert fold_token("raaj") == fold_token("raj")
    # Aspirates
    assert fold_token("bharat") == fold_token("barat")
    assert fold_token("sidharth") == fold_token("sidart")
    # Consonants
    assert fold_token("vishnu") == fold_token("visnu")
    # Schwa deletion
    assert fold_token("ganesha") == fold_token("ganesh")
    # Word end y -> i
    assert fold_token("shanty") == fold_token("santi")


def test_legal_form_extraction():
    core, form = extract_legal_form("Tata Motors Pvt Ltd")
    assert form == "PRIVATE_LIMITED"
    assert "pvt ltd" not in core.lower()

    core2, form2 = extract_legal_form("Google LLC")
    assert form2 == "LLC"
    assert "llc" not in core2.lower()

    # Mismatch conflict
    eq, conflict = check_legal_form_compatibility("PRIVATE_LIMITED", "LLC")
    assert eq == 0 and conflict == 1

    # Match
    eq, conflict = check_legal_form_compatibility("LLC", "LLC")
    assert eq == 1 and conflict == 0


def test_dba_extraction():
    prim, alts = extract_dba_aliases("Reliance Retail (Smart Bazaar)")
    assert "reliance retail" in prim
    assert "smart bazaar" in alts

    prim2, alts2 = extract_dba_aliases("Apex Services dba Cloud Solutions")
    assert "apex services" in prim2
    assert "cloud solutions" in alts2


def test_address_components():
    nums, house, unit = extract_address_numbers("Plot No. 12/3, Sector-5, Suite 200")
    assert "12/3" in nums
    assert "5" in nums
    assert unit == "200"

    h1, h2 = "1600", "1600"
    eq, conflict = compare_house_numbers(h1, h2)
    assert eq == 1 and conflict == 0

    h_diff = "1601"
    eq, conflict = compare_house_numbers(h1, h_diff)
    assert eq == 0 and conflict == 1


def test_postal_code():
    # US ZIP+4 reduction
    assert extract_postal_code("1600 Amphitheatre Pkwy, Mountain View, CA 94043-1351") == "94043"
    # Indian PIN
    assert extract_postal_code("MG Road, Bangalore, Karnataka 560001") == "560001"
    # Prefix similarity
    eq, both, pref = compare_postal_codes("560001", "560002")
    assert eq == 0
    assert both == 1
    assert pref == 5


def test_landmark_extraction():
    clean, landmark = extract_landmark_phrase("Opp. City Mall, MG Road")
    assert "city mall" in landmark.lower()
    assert "mg road" in clean.lower()


def test_abbreviation_alignment():
    # Prefix check
    assert is_abbrev("pvt", "private")
    assert is_abbrev("ltd", "limited")
    assert is_abbrev("corp", "corporation")
    # Alignment score
    score = compute_abbrev_align(["pvt", "ltd"], ["private", "limited"])
    assert score == 1.0


def test_canonicalize_record():
    raw = RawRecord(
        entity_id="S1-00001",
        business_name="Tata Consultancy Services Ltd (TCS)",
        business_address="Opp. Forum Mall, Hosur Road, Bangalore, 560029",
        country="India",
        source="S1",
        split="train",
    )
    can = canonicalize_record(raw)
    assert can.legal_form == "LIMITED"
    assert "tcs" in can.name_alts or can.name_acronym == "tcs"
    assert can.postal == "560029"
    assert can.landmark != ""
    assert can.country_norm == "india"
