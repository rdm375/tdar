import jax.numpy as jnp
import numpy as np
import pytest
from tdar import TDARConfig, farthest_point, sample

def quadratic(x): return 0.8*x[0]**2 + 0.3*x[0]*x[1] + 1.2*x[1]**2 + 0.1*x[0]
def affine(x): return 2*x[0]-3*x[1]+1

def test_exact_budget_and_determinism():
    x=farthest_point(16,candidates=256,seed=7)
    a=sample(quadratic,x,40,TDARConfig(fill_candidates=256,seed=7))
    b=sample(quadratic,x,40,TDARConfig(fill_candidates=256,seed=7))
    assert a.points.shape==(40,2)
    assert np.array_equal(a.points,b.points)

def test_rejects_bad_budget():
    x=farthest_point(16,candidates=64)
    with pytest.raises(ValueError): sample(quadratic,x,8)

def test_affine_still_completes_budget():
    x=farthest_point(16,candidates=128)
    r=sample(affine,x,24,TDARConfig(fill_candidates=128))
    assert len(r.points)==24
