"""Minimal TDAR run on the curved-ridge demonstration problem."""
from problems import curved_ridge
from tdar import farthest_point, sample, jax_oracle

initial = farthest_point(16)
result = sample(jax_oracle(curved_ridge), initial, budget=128)
print(f"samples={len(result.x)}, batches={result.iterations}")
