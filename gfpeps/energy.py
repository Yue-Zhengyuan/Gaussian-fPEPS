from jax import vmap
import jax.numpy as jnp
from .cormat import G_fid, Gin_bz
from .cormat import peps_cormat


def energy_function(bz, t=1.0, D1X=0.0, D1Y=0.0, mu=0.0):
    xis = vmap(lambda x: -2 * t * jnp.sum(x) - mu)(jnp.cos(bz))
    Ds = vmap(lambda x: jnp.vdot(x, jnp.array([2 * D1X, 2 * D1Y])))(jnp.cos(bz))

    def energy(Gouts):
        hoppings = jnp.real(xis * (2 - Gouts[:, 0, 1] - Gouts[:, 2, 3]) / 2)
        pairings = (-1 / 2) * jnp.real(
            Ds
            * (
                Gouts[:, 3, 0]
                + Gouts[:, 2, 1]
                + 1.0j * (Gouts[:, 3, 1] - Gouts[:, 2, 0])
            )
        )
        return jnp.mean(hoppings + pairings)

    return energy


def runtime_loss(bz, Nv=2, t=1.0, D1X=0.0, D1Y=0.0, mu=0.0):
    Gins = Gin_bz(bz, Nv)
    energy = energy_function(bz, t, D1X, D1Y, mu)

    def lossT(T):
        Gfid = G_fid(T)
        Gouts = peps_cormat(Gfid, Gins)
        return energy(Gouts)

    return lossT
