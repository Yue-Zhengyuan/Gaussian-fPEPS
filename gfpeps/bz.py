import jax.numpy as jnp


def BrillouinZone(Lx: int, Ly: int):
    r"""
    Create the Brillouin zone for Lx x Ly square lattice,
    with anti-PBC and PBC on the x and y directions.
    """
    X, Y = jnp.meshgrid((jnp.arange(Lx) - 0.5) / Lx, (jnp.arange(Ly)) / Ly)
    G = 2 * jnp.pi * jnp.array([X.flatten(), Y.flatten()]).T
    return G
