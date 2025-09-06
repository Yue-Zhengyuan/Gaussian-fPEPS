from jax import vmap
import jax.numpy as jnp
from jax.scipy.linalg import block_diag, inv

# ---- Virtual (input) state ----


def Gin_block(k):
    r"""Create a single G_{in} along one direction"""
    t = jnp.exp(1j * k)
    ct = -jnp.exp(-1j * k)
    return jnp.array([[0, 0, 0, ct], [0, 0, ct, 0], [0, t, 0, 0], [t, 0, 0, 0]])


def Gin_dim(k, Nv: int):
    r"""Create G_{in} along a direction"""
    tmp = Gin_block(k)
    return block_diag(*[tmp for i in range(Nv)])


def Gin_k(k, Nv: int):
    r"""Create Gamma In for k for Nv times -> Gamma In all direction
    The order follow order in input k
    """
    return block_diag(*[Gin_dim(ki, Nv) for ki in k])


def Gin_bz(bz, Nv: int):
    r"""Get the virtual correlation matrix for each k in the Brillouin zone"""

    def _Gin_Nv(k):
        return Gin_k(k, Nv)

    return vmap(_Gin_Nv)(bz)


# ---- Fidicual (local) state ----


def G_fid(T: jnp.ndarray):
    r"""Get fiducial state correlation matrix from orthogonal matrix T"""
    n2, m2 = T.shape
    assert n2 == m2 and n2 % 2 == 0
    U = T[0::2, :]
    V = T[1::2, :]
    return U.T @ V - V.T @ U


def getABD(G: jnp.ndarray):
    r"""get A,B,D blocks from slicing the fiducial state correlation matrix"""
    A = G[0:4, 0:4]
    B = G[0:4, 4:]
    D = G[4:, 4:]
    return A, B, D


# ---- PEPS (output) state ----


def peps_cormat(Gfid, Gin):
    r"""
    Given the fiducial state correlation matrix `Gfid`,
    and the Fourier component of the virtual state correlation matrix `Gin`,
    calculate the Fourier component of the PEPS correlation matrix.
    """
    A, B, D = getABD(Gfid)
    return A + B @ inv(D + Gin) @ jnp.transpose(B)
