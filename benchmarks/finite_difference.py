"""Finite-difference first-order oracles with explicit function-call accounting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np


@dataclass
class EvaluationCounter:
    """Count scalar function evaluations without changing the function interface."""
    function: Callable[[np.ndarray], float]
    calls: int = 0

    def __call__(self, x: np.ndarray) -> float:
        self.calls += 1
        return float(np.asarray(self.function(np.asarray(x, float))))


def finite_difference_oracle(
    function: Callable[[np.ndarray], float],
    *,
    scheme: Literal["forward", "central"] = "forward",
    step: float = 1e-5,
    lower=(0.0, 0.0),
    upper=(1.0, 1.0),
):
    """Return ``(oracle, counter)`` using bounded finite differences.

    The forward scheme costs exactly 3 scalar function calls per sampled
    location in 2-D (base value plus one perturbation per coordinate). The
    central scheme costs exactly 5 calls (base value plus two perturbations per
    coordinate). Near a boundary, inward one-sided formulas preserve those
    fixed costs and never evaluate outside ``[lower, upper]``.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    if scheme not in {"forward", "central"}:
        raise ValueError("scheme must be 'forward' or 'central'")

    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)
    if lo.shape != (2,) or hi.shape != (2,) or np.any(lo >= hi):
        raise ValueError("lower and upper must define a 2-D box")
    if np.any(step > (hi - lo) / 2):
        raise ValueError("step is too large for the domain")

    counter = EvaluationCounter(function)

    def oracle(x):
        x = np.asarray(x, float)
        if x.shape != (2,) or np.any(x < lo) or np.any(x > hi):
            raise ValueError("point lies outside the finite-difference domain")
        f0 = counter(x)
        grad = np.empty(2, float)
        for j in range(2):
            e = np.zeros(2); e[j] = step
            if scheme == "forward":
                if x[j] + step <= hi[j]:
                    grad[j] = (counter(x + e) - f0) / step
                else:
                    grad[j] = (f0 - counter(x - e)) / step
            else:
                if x[j] - step >= lo[j] and x[j] + step <= hi[j]:
                    grad[j] = (counter(x + e) - counter(x - e)) / (2 * step)
                elif x[j] + 2 * step <= hi[j]:
                    grad[j] = (-3*f0 + 4*counter(x + e) - counter(x + 2*e)) / (2*step)
                else:
                    grad[j] = (3*f0 - 4*counter(x - e) + counter(x - 2*e)) / (2*step)
        return f0, grad

    oracle.calls_per_location = 3 if scheme == "forward" else 5
    oracle.scheme = scheme
    oracle.step = step
    return oracle, counter
