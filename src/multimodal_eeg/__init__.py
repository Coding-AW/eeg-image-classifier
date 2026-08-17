"""Public interfaces for the NICE-style averaged-EEG study."""

from .aligned_comparison import run_aligned_comparison
from .config import ExperimentConfig
from .nice_publication import publish_nice_study

__all__ = ["ExperimentConfig", "publish_nice_study", "run_aligned_comparison"]
