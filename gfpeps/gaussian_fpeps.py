import jax
import numpy as np
import jax.numpy as jnp
from jax import jit
from .bz import BrillouinZone
from .energy import runtime_loss
from .loadwrite import initialT, savelog
from .exact import energy_exact, solve_mu
import logging

import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient

jax.config.update("jax_enable_x64", True)


def gaussian_fpeps(cfg):
    # unpack cfg
    np.random.seed(cfg.params.seed)
    Nv = cfg.params.Nv
    Lx, Ly = cfg.lattice.Lx, cfg.lattice.Ly
    LoadKey, WriteKey = cfg.file.LoadFile, cfg.file.WriteFile

    cfgh = cfg.hamiltonian
    t = cfgh.t
    D1X, D1Y = cfgh.D1X, cfgh.D1Y
    delta, mu = cfgh.delta, cfgh.mu

    Tsize = 8 * Nv + 4
    T = initialT(LoadKey, Tsize)
    U, S, V = np.linalg.svd(T)
    T = U @ V

    bz = BrillouinZone(Lx, Ly)

    if cfgh.solve_mu_from_delta:
        logging.info("Solving mu for given doping...")
        mu = solve_mu(bz, delta, t, D1X, D1Y)

    lossT = jit(runtime_loss(bz, Nv, t, D1X, D1Y, mu))

    def egrad(x):
        return np.array(jax.grad(lossT)(jnp.array(x)))

    @jax.jit
    def hvp_loss(primals, tangents):
        return jax.jvp(jax.grad(lossT), primals, tangents)[1]

    def ehessa(x, v):
        return np.array(hvp_loss((jnp.array(x),), (jnp.array(v),)))

    Eg = energy_exact(bz, t, D1X, D1Y, mu)  # Will use solved mu to calculate Eg
    logging.info("E_exact = {}\n".format(Eg))

    # Optimizer

    manifold = Stiefel(Tsize, Tsize)

    @pymanopt.function.numpy(manifold)
    def cost(x):
        return lossT(x)

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(x):
        return egrad(x)

    @pymanopt.function.numpy(manifold)
    def euclidean_hessian(x, y):
        return ehessa(x, y)

    problem = Problem(
        manifold=manifold,
        cost=cost,
        euclidean_gradient=euclidean_gradient,
        euclidean_hessian=euclidean_hessian,
    )
    solver = ConjugateGradient(log_verbosity=1, max_iterations=cfg.optimizer.MaxIter)

    result = solver.run(problem, initial_point=T)
    log_cost = np.array(result.log["iterations"]["cost"])
    log_gnorm = np.array(result.log["iterations"]["gradient_norm"])

    logging.info("Iterations \t E_peps \t Gradient Norm")
    for iter in range(len(log_cost)):
        logging.info(f"{iter} \t {log_cost[iter]} \t {log_gnorm[iter]}")

    logging.info(
        f"Optimization done!, final cost: {result.cost}, gnorm: {result.gradient_norm }"
    )

    Xopt = result.point
    args = {
        "mu": mu,
        "D1X": D1X,
        "D1Y": D1Y,
        "doping": delta,
        "t": t,
        "Lx": Lx,
        "Ly": Ly,
        "Nv": Nv,
        "seed": cfg.params.seed,
    }
    savelog(WriteKey, Xopt, lossT(Xopt), Eg, args)

    if cfg.file.SaveEachSteps:
        for iter in range(len(log_cost)):
            Xopt = np.array(result.log["iterations"]["point"])[iter]
            savelog(
                WriteKey[:-3] + f"-iter{iter}" + WriteKey[-3:],
                Xopt,
                lossT(Xopt),
                Eg,
                args,
            )
    return Xopt
