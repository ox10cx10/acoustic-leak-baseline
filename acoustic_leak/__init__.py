"""Acoustic features and a grouped logistic-regression baseline."""
from .features import AcousticFeatures, extract_features
from .model import cross_validate, evaluate, make_pipeline

__all__ = ["AcousticFeatures", "extract_features", "make_pipeline", "cross_validate", "evaluate"]
__version__ = "0.1.0"
