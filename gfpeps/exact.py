import jax
import jax.numpy as jnp
from scipy.optimize import root_scalar


def xi_k(k, t, mu):
    return -2 * t * (jnp.cos(k[0]) + jnp.cos(k[1])) - mu


def Delta_k(k, D1X, D1Y):
    return 2 * D1X * jnp.cos(k[0]) + 2 * D1Y * jnp.cos(k[1])


def bogoliubov_E(k, t, D1X, D1Y, mu):
    r"Energy dispersion of Bogoliubov quasi-particles"
    xi = xi_k(k, t, mu)
    D = Delta_k(k, D1X, D1Y)
    return jnp.sqrt(xi**2 + D**2)


def energy_exact(bz, t, D1X, D1Y, mu):
    def f(k):
        return xi_k(k, t, mu) - bogoliubov_E(k, t, D1X, D1Y, mu)

    return jnp.mean(jax.vmap(f)(bz))


def doping_k(k, t, D1X, D1Y, mu):
    xi = xi_k(k, t, mu)
    E = bogoliubov_E(k, t, D1X, D1Y, mu)
    return xi / E


def doping_exact(bz, t, D1X, D1Y, mu):
    return jnp.mean(jax.vmap(lambda k: doping_k(k, t, D1X, D1Y, mu))(bz))


def solve_mu(bz, doping, t, D1X, D1Y, mu_range=(-1.0, 10.0)):
    def f(mu):
        return doping_exact(bz, t, D1X, D1Y, mu) - doping

    return root_scalar(f, bracket=mu_range).root
