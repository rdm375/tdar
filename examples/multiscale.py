from problems import multiscale
from tdar import farthest_point, sample, jax_oracle
result = sample(jax_oracle(multiscale), farthest_point(16), budget=128)
print(result.x.shape, result.y.shape)
