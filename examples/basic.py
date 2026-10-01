import jax.numpy as jnp
from tdar import farthest_point, sample, jax_oracle

def f(x):
    return jnp.exp(-40.0*((x[0]-.35)**2+(x[1]-.6)**2))

r=sample(jax_oracle(f), farthest_point(16), 64)
print(f"samples={len(r.points)}, iterations={r.iterations}")
