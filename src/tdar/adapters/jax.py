"""Optional JAX adapter for TDAR first-order oracles."""
from __future__ import annotations

from typing import Callable


def jax_oracle(f: Callable) -> Callable:
    """Wrap a JAX-differentiable scalar function as a TDAR first-order oracle."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("jax_oracle requires the optional 'jax' extra") from exc

    value_and_grad = jax.value_and_grad(f)

    def oracle(x):
        return value_and_grad(jnp.asarray(x))

    return oracle
