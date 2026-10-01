"""Smallest useful user-defined-function example."""
import jax.numpy as jnp
from tdar import farthest_point, sample


def expensive_function(x):
    return jnp.sin(6.0 * x[0]) * jnp.exp(-3.0 * (x[1] - 0.4)**2)

result = sample(expensive_function, farthest_point(16), budget=80)
X, y = result.x, result.y
print(X.shape, y.shape)
