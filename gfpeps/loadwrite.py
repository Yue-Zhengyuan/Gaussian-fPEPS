import numpy as np
import jax.numpy as jnp
import h5py, os
import logging


def initialT(loadfile, Tsize):
    if loadfile != None and os.path.isfile(loadfile):
        logging.info(f"Try to initialize T from {loadfile}...")
        with h5py.File(loadfile, "r") as f:
            if "/transformer/T" in f.keys():
                T = f["/transformer/T"][:]
                f.close()
                return jnp.reshape(T, (Tsize, Tsize))
            else:
                logging.debug(
                    f"No /transformer/T in {loadfile} found. Switch to random initialization."
                )
                f.close()
                return jnp.array(np.random.rand(Tsize, Tsize))
    return jnp.array(np.random.rand(Tsize, Tsize))


def savelog(writefile, x, E_peps, E_exact, args):
    if writefile != None:
        logging.info(f"Save T to {writefile}")
        with h5py.File(writefile, "w") as f:
            f["/transformer/T"] = x

            f["/energy/E_peps"] = E_peps
            f["/energy/E_exact"] = E_exact

            f["/model/mu"] = args["mu"]
            f["/model/D1X"] = args["D1X"]
            f["/model/D1Y"] = args["D1Y"]
            f["/model/t"] = args["t"]
            f["/model/Nv"] = args["Nv"]
            f["/model/seed"] = args["seed"]
            f["/model/Lx"] = args["Lx"]
            f["/model/Ly"] = args["Ly"]
            f["/model/doping"] = args["doping"]

            f.close()
