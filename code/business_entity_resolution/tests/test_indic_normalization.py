"""Tests for Indic transliteration and cross-lingual normalization."""
from ber.normalization.indic import transliterate_indic, has_indic_characters
from ber.normalization.name import normalize_business_name
from ber.extraction.legal_form import extract_legal_form


def test_has_indic_characters():
    assert has_indic_characters("अल्फा आईटी")
    assert has_indic_characters("నార్త్ సౌత్ ఎస్టేట్")
    assert has_indic_characters("సూర్య ఎక్స్‌పೋರ್ట్స్")
    assert not has_indic_characters("Alpha IT")
    assert not has_indic_characters("123 Main Street")


def test_transliterate_indic():
    assert "alpha" in transliterate_indic("अल्फा आईटी")
    assert "praivet limited" in transliterate_indic("प्राइवेट लिमिटेड")
    assert "praivet limited" in transliterate_indic("ప్రైవేట్ లిమిటెడ్")
    assert "limirrd" in transliterate_indic("ലിമിറ്റഡ്")


def test_indic_legal_form_extraction():
    core, form = extract_legal_form("praim inphrastrkcr praivet limited")
    assert form == "PRIVATE_LIMITED"
    assert "praivet limited" not in core

    core2, form2 = extract_legal_form("shkti phud elelpi")
    assert form2 == "LLP"
    assert "elelpi" not in core2
