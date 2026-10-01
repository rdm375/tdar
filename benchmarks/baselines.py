"""Space-filling baseline designs for TDAR benchmarks.

All designs include the four corners of the unit square. This keeps the convex
hull identical across methods, so piecewise-linear interpolation is evaluated
on the same domain rather than penalizing a method for extrapolation.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import qmc

CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def _with_corners(interior: np.ndarray, n: int) -> np.ndarray:
    if n < 4:
        raise ValueError("benchmark designs require n >= 4")
    return np.vstack([CORNERS, np.asarray(interior, dtype=float)[: n - 4]])


def random_design(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _with_corners(rng.random((max(0, n - 4), 2)), n)


def lhs_design(n: int, seed: int) -> np.ndarray:
    m = max(0, n - 4)
    interior = qmc.LatinHypercube(d=2, seed=seed).random(m) if m else np.empty((0, 2))
    return _with_corners(interior, n)


def sobol_design(n: int) -> np.ndarray:
    """Deterministic unscrambled Sobol design, plus common domain corners."""
    m = max(0, n - 4)
    if not m:
        return CORNERS.copy()
    # Generate a power-of-two block, then take the required prefix. Avoid the
    # all-zero first Sobol point because (0, 0) is already a common corner.
    power = max(1, math.ceil(math.log2(m + 1)))
    interior = qmc.Sobol(d=2, scramble=False).random_base2(power)[1 : m + 1]
    return _with_corners(interior, n)
