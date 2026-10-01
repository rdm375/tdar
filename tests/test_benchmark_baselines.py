import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from baselines import CORNERS, lhs_design, random_design, sobol_design
from metrics import error_metrics, piecewise_linear_predict


def test_baselines_have_requested_size_and_common_corners():
    for design in (random_design(24, 7), lhs_design(24, 7), sobol_design(24)):
        assert design.shape == (24, 2)
        assert np.array_equal(design[:4], CORNERS)
        assert np.all((design >= 0.0) & (design <= 1.0))


def test_piecewise_linear_reproduces_affine_function():
    x = sobol_design(32)
    y = 2.0 * x[:, 0] - 3.0 * x[:, 1] + 1.5
    q = np.random.default_rng(4).random((200, 2))
    pred = piecewise_linear_predict(x, y, q)
    truth = 2.0 * q[:, 0] - 3.0 * q[:, 1] + 1.5
    assert error_metrics(truth, pred)["rmse"] < 1e-12
