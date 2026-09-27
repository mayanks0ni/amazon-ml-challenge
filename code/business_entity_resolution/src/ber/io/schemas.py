"""Data structures and schemas for business entity resolution."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class RawRecord:
    """Raw record as read directly from TSV."""
    entity_id: str
    business_name: str
    business_address: str
    country: str
    source: str = ""  # 'S1', 'S2', or 'S3'
    split: str = "train"  # 'train' or 'test'


@dataclass
class CanonicalRecord:
    """Normalized canonical representation of a business record."""
    split: str
    source: str  # 'S1', 'S2', 'S3'
    entity_id: str
    name_raw: str
    addr_raw: str
    country_raw: str

    # Name representations
    name_norm: str = ""
    name_core: str = ""
    name_alts: List[str] = field(default_factory=list)
    legal_form: str = ""
    name_fold: str = ""
    name_acronym: str = ""
    name_tokens: List[str] = field(default_factory=list)
    name_core_tokens: List[str] = field(default_factory=list)
    name_clean: str = ""

    # Address representations
    addr_norm: str = ""
    addr_tokens: List[str] = field(default_factory=list)
    addr_fold: str = ""
    addr_nums: List[str] = field(default_factory=list)
    addr_clean: str = ""
    sorted_addr: str = ""
    postal: str = ""
    house_no: str = ""
    unit: str = ""
    landmark: str = ""
    addr_tail: str = ""

    # Country
    country_norm: str = ""

    # Missing flags
    name_missing: bool = False
    addr_missing: bool = False
    postal_missing: bool = False
    house_no_missing: bool = False


@dataclass
class CandidateProvenance:
    """Provenance tracking for a candidate pair."""
    mask: int = 0
    rank: Dict[str, int] = field(default_factory=dict)
    score: Dict[str, float] = field(default_factory=dict)
