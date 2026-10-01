"""Public API for TDAR."""
from .core import TDARConfig, TDARResult, TDARStep, sample
from .designs import farthest_point

__all__ = ["TDARConfig", "TDARResult", "TDARStep", "sample", "farthest_point"]
__version__ = "0.2.0.dev0"
