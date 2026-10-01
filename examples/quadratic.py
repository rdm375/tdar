from problems import quadratic
from tdar import farthest_point, sample, jax_oracle
result = sample(jax_oracle(quadratic), farthest_point(16), budget=64)
print(result.x.shape, result.y.shape)
