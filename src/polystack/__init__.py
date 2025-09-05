"""
polystack
=========

Public API surface for the polystack package.
"""

from __future__ import annotations

try:
    from ._version import __version__
except Exception:
    __version__ = "0.0.0"

# Re-export public functions
from .crossval import oof_by_view
from .estimator import Polystack
from .meta_features import MetaFeatures, register, create

__all__ = [
    "oof_by_view",
    "Polystack",
    "MetaFeatures",
    "register",
    "create",
    "__version__",
]

