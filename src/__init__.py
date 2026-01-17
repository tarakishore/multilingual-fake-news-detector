"""
Multilingual Fake News Detection with Explainability
Main package initialization.
"""

__version__ = "1.0.0"
__author__ = "Multilingual Fake News Detection Team"

from .models.classifier import FakeNewsClassifier
from .explainability.integrated_gradients import IntegratedGradientsExplainer

__all__ = [
    "FakeNewsClassifier",
    "IntegratedGradientsExplainer",
]
