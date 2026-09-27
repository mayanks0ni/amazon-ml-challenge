"""Canonical pipeline converting RawRecord into full CanonicalRecord."""
from ber.io.schemas import RawRecord, CanonicalRecord
from ber.normalization.name import normalize_business_name
from ber.normalization.address import normalize_address
from ber.normalization.country import normalize_country
from ber.normalization.indic import transliterate_indic, has_indic_characters


def canonicalize_record(raw: RawRecord) -> CanonicalRecord:
    """Transforms a RawRecord into a fully structured, multi-representation CanonicalRecord."""
    country_norm = normalize_country(raw.country)
    # Transliterate Indic scripts to Latin phonetic characters
    trans_name = transliterate_indic(raw.business_name)
    trans_addr = transliterate_indic(raw.business_address)

    name_dict = normalize_business_name(trans_name)
    addr_dict = normalize_address(trans_addr, country=country_norm)

    # If original name was in Indic script, keep original raw name in name_alts as well
    name_alts = list(name_dict["name_alts"])
    if has_indic_characters(raw.business_name) and raw.business_name not in name_alts:
        name_alts.append(raw.business_name)

    return CanonicalRecord(
        split=raw.split,
        source=raw.source,
        entity_id=raw.entity_id,
        name_raw=raw.business_name,
        addr_raw=raw.business_address,
        country_raw=raw.country,
        # Name fields
        name_norm=name_dict["name_norm"],
        name_core=name_dict["name_core"],
        name_alts=name_alts,
        legal_form=name_dict["legal_form"],
        name_fold=name_dict["name_fold"],
        name_acronym=name_dict["name_acronym"],
        name_tokens=name_dict["name_tokens"],
        name_core_tokens=name_dict["name_core_tokens"],
        name_clean=name_dict.get("name_clean", ""),
        # Address fields
        addr_norm=addr_dict["addr_norm"],
        addr_tokens=addr_dict["addr_tokens"],
        addr_fold=addr_dict["addr_fold"],
        addr_nums=addr_dict["addr_nums"],
        addr_clean=addr_dict.get("addr_clean", ""),
        sorted_addr=addr_dict.get("sorted_addr", ""),
        postal=addr_dict["postal"],
        house_no=addr_dict["house_no"],
        unit=addr_dict["unit"],
        landmark=addr_dict["landmark"],
        addr_tail=addr_dict["addr_tail"],
        # Country
        country_norm=country_norm,
        # Missing flags
        name_missing=name_dict["name_missing"],
        addr_missing=addr_dict["addr_missing"],
        postal_missing=addr_dict["postal_missing"],
        house_no_missing=addr_dict["house_no_missing"],
    )
