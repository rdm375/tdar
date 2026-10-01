"""Surrogate fitting and held-out metrics used by the benchmark harness."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import LinearNDInterpolator


def piecewise_linear_predict(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Fit a Delaunay piecewise-linear interpolant and evaluate ``query``."""
    pred = np.asarray(LinearNDInterpolator(np.asarray(x), np.asarray(y))(query), dtype=float)
    if np.isnan(pred).any():
        raise RuntimeError("held-out points fell outside the design convex hull")
    return pred


def error_metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(pred) - np.asarray(truth)
    return {
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
    }
