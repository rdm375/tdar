"""Public API for TDAR."""
from .core import TDARConfig, TDARResult, TDARStep, sample
from .designs import farthest_point
from .interchange import save_simplicial_cpwa
from .adapters.jax import jax_oracle

__all__ = ["TDARConfig", "TDARResult", "TDARStep", "sample", "farthest_point", "jax_oracle", "save_simplicial_cpwa"]
__version__ = "0.2.0.dev0"
