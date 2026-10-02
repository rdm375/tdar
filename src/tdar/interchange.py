"""Interchange helpers for exporting TDAR's final simplicial CPWA surrogate."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .core import TDARResult


FORMAT_NAME = "simplicial-cpwa"
FORMAT_VERSION = 1


def save_simplicial_cpwa(
    result: TDARResult,
    path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Export the final TDAR piecewise-linear interpolant as a portable NPZ.

    The artifact contains only the data needed to reconstruct the scalar
    simplicial CPWA function: ``points``, ``simplices``, and ``values``.
    TDAR gradients are intentionally excluded because they guide refinement but
    are not part of the final piecewise-linear surrogate.

    User metadata, when supplied, is written to ``<artifact>.json`` rather than
    embedded in the NPZ so the numerical interchange artifact remains simple
    and loadable without pickle support.
    """
    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("simplicial CPWA artifact path must end in .npz")

    points = np.asarray(result.points, dtype=np.float64)
    values = np.asarray(result.values, dtype=np.float64).reshape(-1)
    simplices = np.asarray(result.simplices, dtype=np.int64)

    _validate_export_arrays(points, simplices, values)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, points=points, simplices=simplices, values=values)

    if metadata is not None:
        sidecar = destination.with_suffix(".json")
        document = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "producer": "tdar",
            "points": int(points.shape[0]),
            "simplices": int(simplices.shape[0]),
            "dimension": int(points.shape[1]),
            "metadata": dict(metadata),
        }
        sidecar.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return destination


def _validate_export_arrays(points, simplices, values) -> None:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("TDAR points must have shape (n, 2)")
    if values.shape != (len(points),):
        raise ValueError("TDAR values must contain one scalar per point")
    if simplices.ndim != 2 or simplices.shape[1] != 3:
        raise ValueError("TDAR simplices must have shape (m, 3)")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
        raise ValueError("TDAR export contains non-finite points or values")
    if simplices.size and (np.min(simplices) < 0 or np.max(simplices) >= len(points)):
        raise ValueError("TDAR simplex index is outside the point array")
