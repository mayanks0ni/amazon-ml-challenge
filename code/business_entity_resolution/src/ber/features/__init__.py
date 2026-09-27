"""Feature extraction and engineering module."""
from ber.features.name import extract_name_features
from ber.features.address import extract_address_features
from ber.features.cross import extract_cross_features
from ber.features.rarity import CorpusIDF, compute_idf_shared_unshared
from ber.features.group import compute_group_context_features
from ber.features.registry import (
    extract_features_for_candidates,
    build_pair_feature_vector,
    get_monotone_constraints,
)

__all__ = [
    "extract_name_features",
    "extract_address_features",
    "extract_cross_features",
    "CorpusIDF",
    "compute_idf_shared_unshared",
    "compute_group_context_features",
    "extract_features_for_candidates",
    "build_pair_feature_vector",
    "get_monotone_constraints",
]
