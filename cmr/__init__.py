"""
CMR (Confusion Matrix Rules) - A rule-based explainable AI method for binary classification
"""

from .core import make_cmc, get_exclusive_rules, get_minmax_rules
from .metrics import compute_rule_coverage, compute_rule_overlap
from .inference import apply_rules

__version__ = "1.0.0"

__all__ = [
    'make_cmc',
    'get_exclusive_rules',
    'get_minmax_rules',
    'compute_rule_coverage',
    'compute_rule_overlap',
    'apply_rules'
]
