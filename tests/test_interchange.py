import json

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from tdar import TDARConfig, farthest_point, sample, save_simplicial_cpwa


def quadratic_oracle(x):
    value = 0.8*x[0]**2 + 0.3*x[0]*x[1] + 1.2*x[1]**2 + 0.1*x[0]
    gradient = np.array([1.6*x[0] + 0.3*x[1] + 0.1, 0.3*x[0] + 2.4*x[1]])
    return value, gradient


def result():
    x = farthest_point(16, candidates=128, seed=7)
    return sample(quadratic_oracle, x, 24, TDARConfig(fill_candidates=128, seed=7))


def test_result_simplices_are_final_delaunay_triangles():
    r = result()
    simplices = r.simplices
    assert simplices.ndim == 2
    assert simplices.shape[1] == 3
    assert simplices.dtype == np.int64
    assert np.min(simplices) >= 0
    assert np.max(simplices) < len(r.points)


def test_export_contains_generic_simplicial_cpwa_arrays(tmp_path):
    r = result()
    path = save_simplicial_cpwa(r, tmp_path / "surrogate.npz")

    with np.load(path, allow_pickle=False) as artifact:
        assert set(artifact.files) == {"points", "simplices", "values"}
        assert np.array_equal(artifact["points"], r.points)
        assert np.array_equal(artifact["simplices"], r.simplices)
        assert np.array_equal(artifact["values"], r.values)
        assert artifact["points"].dtype == np.float64
        assert artifact["simplices"].dtype == np.int64
        assert artifact["values"].dtype == np.float64


def test_export_preserves_final_piecewise_linear_interpolant(tmp_path):
    r = result()
    path = save_simplicial_cpwa(r, tmp_path / "surrogate.npz")

    with np.load(path, allow_pickle=False) as artifact:
        points = artifact["points"]
        simplices = artifact["simplices"]
        values = artifact["values"]

    native = LinearNDInterpolator(r.points, r.values)

    # Evaluate the exported simplicial function directly using barycentric
    # coordinates, independently of a fresh Delaunay triangulation.
    rng = np.random.default_rng(20261002)
    for simplex in simplices:
        weights = rng.dirichlet(np.ones(3), size=25)
        x = weights @ points[simplex]
        exported = weights @ values[simplex]
        np.testing.assert_allclose(exported, native(x), atol=2e-12, rtol=0)


def test_metadata_is_optional_json_sidecar(tmp_path):
    r = result()
    path = save_simplicial_cpwa(
        r,
        tmp_path / "methane-z.npz",
        metadata={"target": "methane-Z", "normalization": "unit-square"},
    )

    sidecar = path.with_suffix(".json")
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    assert document["format"] == "simplicial-cpwa"
    assert document["format_version"] == 1
    assert document["producer"] == "tdar"
    assert document["dimension"] == 2
    assert document["metadata"]["target"] == "methane-Z"


def test_export_rejects_non_npz_suffix(tmp_path):
    r = result()
    try:
        save_simplicial_cpwa(r, tmp_path / "surrogate.bin")
    except ValueError as exc:
        assert ".npz" in str(exc)
    else:
        raise AssertionError("expected ValueError")
