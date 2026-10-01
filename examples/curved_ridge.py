"""Minimal TDAR run on the curved-ridge demonstration problem."""
from problems import curved_ridge
from tdar import farthest_point, sample

initial = farthest_point(16)
result = sample(curved_ridge, initial, budget=128)
print(f"samples={len(result.x)}, batches={result.iterations}")
