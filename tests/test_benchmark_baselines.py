import sys
from pathlib import Path
import numpy as np
import pytest

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


def test_finite_difference_oracles_have_exact_cost_and_stay_in_domain():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from finite_difference import finite_difference_oracle

    seen=[]
    def f(x):
        x=np.asarray(x,float); seen.append(x.copy())
        return x[0]**2 + 3*x[1]**2

    for scheme,cost,tol in (("forward",3,2e-3),("central",5,1e-8)):
        for x in (np.array([0.,0.]),np.array([.4,.6]),np.array([1.,1.])):
            seen.clear(); oracle,counter=finite_difference_oracle(f,scheme=scheme,step=1e-4)
            value,g=oracle(x)
            assert counter.calls==cost
            assert all(np.all((p>=0)&(p<=1)) for p in seen)
            assert value == pytest.approx(x[0]**2+3*x[1]**2)
            assert np.allclose(g,[2*x[0],6*x[1]],atol=tol)
