"""Small, readable demonstration problems used by TDAR examples."""
import jax.numpy as jnp


def quadratic(x):
    """Smooth anisotropic quadratic."""
    x0, x1 = x
    return x0**2 + 4.0 * x1**2


def curved_ridge(x):
    """Narrow curved transition embedded in a simple background."""
    x0, x1 = x
    center = 0.55 + 0.15 * jnp.sin(2.0 * jnp.pi * x0)
    return 0.2 * x0 + 0.1 * x1 + jnp.tanh(20.0 * (x1 - center))


def multiscale(x):
    """Broad sinusoidal structure plus a narrow localized feature."""
    x0, x1 = x
    broad = jnp.sin(2.0 * jnp.pi * x0) * jnp.sin(2.0 * jnp.pi * x1)
    narrow = 0.2 * jnp.exp(-((x0 - 0.72)**2 + (x1 - 0.31)**2) / 0.002)
    return broad + narrow
