import sys

import numpy as np
import pytest

from tdar.adapters.thermogpu import ThermoGPUDomain, ThermoGPUOracle


FAKE = r'''import sys
mode = sys.argv[1]
assert sys.argv[2] == "--stdin"
for line in sys.stdin:
    t, p = map(float, line.split())
    if mode == "good":
        z = 0.5 + 1e-3*t - 2e-8*p
        print(t, p, z, 1e-3, -2e-8, "one_real", 1e-17, flush=True)
    elif mode == "roots":
        print(t, p, 1.0, 0.0, 0.0, "three_real", 0.0, flush=True)
    elif mode == "bad":
        print("not enough fields", flush=True)
'''


def command(mode):
    return [sys.executable, "-u", "-c", FAKE, mode]


def test_domain_maps_unit_square_and_scales_gradient():
    d = ThermoGPUDomain((250.0, 350.0), (1e6, 10e6))
    assert d.physical([0.5, 4/9]) == pytest.approx((300.0, 5e6))
    np.testing.assert_allclose(d.normalized_gradient(1e-3, -2e-8), [0.1, -0.18])


def test_persistent_oracle_returns_normalized_gradient():
    with ThermoGPUOracle(command("good")) as oracle:
        z1, g1 = oracle(np.array([0.5, 4/9]))
        z2, g2 = oracle(np.array([0.7, 7/9]))
        pid = oracle._process.pid
        assert oracle._process.poll() is None
        assert oracle._process.pid == pid
    assert z1 == pytest.approx(0.7)
    np.testing.assert_allclose(g1, [0.1, -0.18])
    assert np.isfinite(z2)
    np.testing.assert_allclose(g2, [0.1, -0.18])


def test_rejects_unexpected_root_classification():
    with ThermoGPUOracle(command("roots")) as oracle:
        with pytest.raises(RuntimeError, match="three_real"):
            oracle([0.5, 0.5])


def test_rejects_malformed_output():
    with ThermoGPUOracle(command("bad")) as oracle:
        with pytest.raises(RuntimeError, match="malformed"):
            oracle([0.5, 0.5])
