"""Country normalization handling open-set unseen countries (Section 10.5)."""
from ber.normalization.unicode import normalize_unicode

# Safe canonical mapping for common country spelling variants
COUNTRY_ALIASES = {
    "usa": "us",
    "united states": "us",
    "united states of america": "us",
    "u.s.a.": "us",
    "u.s.": "us",
    "in": "india",
    "ind": "india",
    "fr": "france",
    "fra": "france",
}


def normalize_country(raw_country: str) -> str:
    """Normalizes country name while strictly maintaining open-set compliance (Rule O8).

    - Never drops a record due to unknown country.
    - Never coerces unknown countries into a fixed closed set.
    """
    if not raw_country:
        return ""

    c = normalize_unicode(raw_country).strip()
    return COUNTRY_ALIASES.get(c, c)
