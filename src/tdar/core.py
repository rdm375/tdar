"""Frozen TDAR v1 adaptive refinement algorithm for 2-D scalar functions."""
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from scipy.spatial import Delaunay


@dataclass(frozen=True)
class TDARConfig:
    """Configuration for the frozen TDAR method-v1 decision path."""

    batch_size: int = 8
    maximin_order: int = 12
    min_separation_fraction: float = 0.10
    closure_ratio_threshold: float = 1e-2
    fill_candidates: int = 16384
    seed: int = 12345
    duplicate_tolerance: float = 1e-12
    boundary_cutoff_multiplier: float = 100.0
    record_history: bool = False


@dataclass(frozen=True)
class TDARStep:
    """One refinement batch, recorded for diagnostics and visualization.

    History is deliberately opt-in. Arrays are snapshots from immediately before
    ``proposed_points`` were evaluated and appended to the design.
    """

    iteration: int
    points: np.ndarray
    simplices: np.ndarray
    boundary_edges: np.ndarray
    triangle_indicators: np.ndarray
    boundary_indicators: np.ndarray
    proposed_points: np.ndarray
    selected_entities: tuple[tuple[Literal["triangle", "edge", "fill"], int], ...]


@dataclass
class TDARResult:
    """Completed TDAR design and optional refinement history."""

    points: np.ndarray
    values: np.ndarray
    gradients: np.ndarray
    iterations: int
    history: tuple[TDARStep, ...] = field(default_factory=tuple)

    @property
    def x(self) -> np.ndarray:
        """Alias for sample locations, convenient for surrogate libraries."""
        return self.points

    @property
    def y(self) -> np.ndarray:
        """Alias for sampled scalar values."""
        return self.values


def evaluate_oracle(oracle: Callable, points: np.ndarray):
    """Evaluate a first-order oracle point-by-point.

    The oracle must return ``(value, gradient)`` for one point. Keeping this
    interface framework-neutral lets callers supply analytic sensitivities,
    adjoints, autodiff adapters, or other derivative providers.
    """
    values = []
    gradients = []
    for point in np.asarray(points, float):
        value, gradient = oracle(point)
        values.append(float(np.asarray(value)))
        g = np.asarray(gradient, float)
        if g.shape != (2,):
            raise ValueError("oracle gradient must have shape (2,)")
        gradients.append(g)
    return np.asarray(values, float), np.asarray(gradients, float)


def _barycentric_lattice(order):
    out = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            k = order - i - j
            out.append((i / order, j / order, k / order))
    return np.asarray(out, float)


def _triangle_probes(points, simplices):
    tri = np.asarray(points, float)[np.asarray(simplices, int)]
    c = tri.mean(axis=1)
    return np.stack([c, .5 * (tri[:, 0] + tri[:, 1]), .5 * (tri[:, 1] + tri[:, 2]), .5 * (tri[:, 2] + tri[:, 0])], axis=1)


def triangle_indicator(points, values, gradients, simplices):
    points = np.asarray(points, float); values = np.asarray(values).reshape(-1); gradients = np.asarray(gradients); simplices = np.asarray(simplices, int)
    tri = points[simplices]; fv = values[simplices]; gv = gradients[simplices]; probes = _triangle_probes(points, simplices)
    delta = probes[:, :, None, :] - tri[:, None, :, :]
    pred = fv[:, None, :] + np.einsum('tpvi,tvi->tpv', delta, gv)
    d = np.sqrt(np.mean((pred - pred.mean(axis=2, keepdims=True)) ** 2, axis=2))
    return d.max(axis=1)


def boundary_indicator(points, values, gradients, edges):
    points = np.asarray(points, float); values = np.asarray(values).reshape(-1); gradients = np.asarray(gradients); edges = np.asarray(edges, int)
    if not len(edges): return np.empty(0)
    ev = points[edges]; fv = values[edges]; gv = gradients[edges]; t = np.array([.25, .5, .75])
    probes = (1 - t[None, :, None]) * ev[:, 0, None, :] + t[None, :, None] * ev[:, 1, None, :]
    pred = fv[:, None, :] + np.einsum('epvi,evi->epv', probes[:, :, None, :] - ev[:, None, :, :], gv)
    d = np.sqrt(np.mean((pred - pred.mean(axis=2, keepdims=True)) ** 2, axis=2))
    return d.max(axis=1)


def _boundary_edges(d):
    return np.unique(np.sort(np.asarray(d.convex_hull, int), axis=1), axis=0)


def _is_new(p, existing, proposed, tol):
    if np.any(np.linalg.norm(existing - p, axis=1) <= tol): return False
    return not proposed or not np.any(np.linalg.norm(np.asarray(proposed) - p, axis=1) <= tol)


def _interior_maximin(points, simplex, selected, order, min_fraction):
    tri = points[np.asarray(simplex, int)]; cand = _barycentric_lattice(order) @ tri
    ref = np.vstack([points, np.asarray(selected, float)]) if selected else points
    d = np.min(np.linalg.norm(cand[:, None, :] - ref[None, :, :], axis=2), axis=1); j = int(np.argmax(d))
    h = max(np.linalg.norm(tri[0] - tri[1]), np.linalg.norm(tri[1] - tri[2]), np.linalg.norm(tri[2] - tri[0]))
    return cand[j].copy(), bool(d[j] >= min_fraction * h)


def _boundary_maximin(points, edge, selected, order, min_fraction):
    a, b = points[np.asarray(edge, int)]; ab = b - a; h = float(np.linalg.norm(ab)); t = np.arange(1, order, dtype=float) / order
    cand = (1 - t[:, None]) * a + t[:, None] * b; ref = [a, b]
    for q in np.asarray(selected, float) if selected else []:
        u = float(np.dot(q - a, ab) / np.dot(ab, ab)); proj = a + u * ab
        if -1e-12 <= u <= 1 + 1e-12 and np.linalg.norm(q - proj) <= 1e-10 * max(1., h): ref.append(q)
    d = np.min(np.linalg.norm(cand[:, None, :] - np.asarray(ref)[None, :, :], axis=2), axis=1); j = int(np.argmax(d))
    return cand[j].copy(), bool(d[j] >= min_fraction * h)


def _farthest_fill(existing, proposed, count, pool):
    if count <= 0: return []
    selected = np.vstack([existing, np.asarray(proposed)]) if proposed else existing.copy()
    d = np.min(np.linalg.norm(pool[:, None, :] - selected[None, :, :], axis=2), axis=1); out = []
    for _ in range(count):
        j = int(np.argmax(d)); p = pool[j].copy(); out.append(p); d = np.minimum(d, np.linalg.norm(pool - p, axis=1)); d[j] = -np.inf
    return out


def sample(oracle, initial_points, budget, config=TDARConfig()):
    """Run frozen TDAR v1 refinement on the unit square.

    Parameters
    ----------
    oracle:
        Callable accepting one point of shape ``(2,)`` and returning
        ``(value, gradient)``. The gradient must have shape ``(2,)``.
    initial_points:
        Initial design with shape ``(n, 2)``. It should cover the unit square;
        :func:`tdar.farthest_point` is the standard initializer.
    budget:
        Total number of sampled locations, including the initial design.
    config:
        Method configuration. Set ``record_history=True`` for visualization or
        diagnostics; normal sampling avoids this storage overhead.
    """
    points = np.asarray(initial_points, float).copy()
    if points.ndim != 2 or points.shape[1] != 2: raise ValueError('initial_points must have shape (n, 2)')
    if budget < len(points): raise ValueError('budget cannot be smaller than initial sample count')
    if config.batch_size < 1 or config.maximin_order < 2 or config.min_separation_fraction < 0 or config.closure_ratio_threshold < 0: raise ValueError('invalid TDAR configuration')
    values, gradients = evaluate_oracle(oracle, points)
    pool = np.random.default_rng(config.seed).random((config.fill_candidates, 2)); iterations = 0; history = []
    while len(points) < budget:
        remaining = budget - len(points); batch = min(config.batch_size, remaining); d = Delaunay(points); simplices = d.simplices
        ts = triangle_indicator(points, values, gradients, simplices); edges = _boundary_edges(d); es = boundary_indicator(points, values, gradients, edges)
        scale = max(1., float(np.max(np.abs(ts), initial=0.)), float(np.max(np.abs(es), initial=0.)))
        edge_tol = config.boundary_cutoff_multiplier * np.finfo(float).eps * scale
        lookup = {tuple(sorted(map(int, e))): k for k, e in enumerate(edges)}
        entities = [(float(ts[k]), 'triangle', k) for k in range(len(simplices))] + [(float(es[k]), 'edge', k) for k in range(len(edges))]
        entities.sort(key=lambda z: z[0], reverse=True); proposed = []; used = set(); selected_entities = []
        for _, kind, k in entities:
            if kind == 'edge':
                if k in used or es[k] <= edge_tol: continue
                p, ok = _boundary_maximin(points, edges[k], proposed, config.maximin_order, config.min_separation_fraction); used.add(k)
                if not ok or not _is_new(p, points, proposed, config.duplicate_tolerance): continue
                proposed.append(p); selected_entities.append(('edge', int(k)))
            else:
                verts = list(map(int, simplices[k])); closure = []
                for a, b in [(verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])]:
                    ek = lookup.get(tuple(sorted((a, b))))
                    if ek is not None and ek not in used and es[ek] > edge_tol:
                        ratio = float(es[ek]) / float(ts[k]) if ts[k] > 0 else np.inf
                        if ratio >= config.closure_ratio_threshold: closure.append((float(es[ek]), ek))
                accepted = False
                for _, ek in sorted(closure, reverse=True):
                    p, ok = _boundary_maximin(points, edges[ek], proposed, config.maximin_order, config.min_separation_fraction); used.add(ek)
                    if ok and _is_new(p, points, proposed, config.duplicate_tolerance):
                        proposed.append(p); selected_entities.append(('edge', int(ek))); accepted = True; break
                if not accepted:
                    p, ok = _interior_maximin(points, simplices[k], proposed, config.maximin_order, config.min_separation_fraction)
                    if not ok or not _is_new(p, points, proposed, config.duplicate_tolerance): continue
                    proposed.append(p); selected_entities.append(('triangle', int(k)))
            if len(proposed) == batch: break
        fill = _farthest_fill(points, proposed, batch - len(proposed), pool)
        proposed.extend(fill); selected_entities.extend(('fill', -1) for _ in fill)
        new = np.asarray(proposed, float)[:remaining]
        if config.record_history:
            history.append(TDARStep(
                iteration=iterations,
                points=points.copy(), simplices=np.asarray(simplices, int).copy(), boundary_edges=np.asarray(edges, int).copy(),
                triangle_indicators=np.asarray(ts, float).copy(), boundary_indicators=np.asarray(es, float).copy(),
                proposed_points=new.copy(), selected_entities=tuple(selected_entities[:len(new)]),
            ))
        nv, ng = evaluate_oracle(oracle, new); points = np.vstack([points, new]); values = np.concatenate([values, nv]); gradients = np.vstack([gradients, ng]); iterations += 1
    return TDARResult(points, values, gradients, iterations, tuple(history))
