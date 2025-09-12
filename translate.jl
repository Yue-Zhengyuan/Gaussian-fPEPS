include("imports.jl")
cd(@__DIR__)

function main_translate(folder::AbstractString, χenv::Int)
    data = h5open(folder * "/data.h5", "r")
    U = Matrix(data["transformer/T"][]')
    Np, Nv = 2, data["model/Nv"][]

    tps_dir = folder * "noproj/"
    mkpath(tps_dir)
    logs = open(tps_dir * "translate_log.txt", "w")
    redirect_stdout(logs)
    redirect_stderr(logs)

    @info "Translating PEPS..."
    peps = translate(U, Np, Nv)
    save_iPEPS(tps_dir, peps)
    @info "PEPS saved to $(tps_dir)"

    @info "Converging CTMRG with χenv = 4..."
    Espace = Vect[FermionParity](0 => 4, 1 => 4)
    env = CTMRGEnv(randn, ComplexF64, peps, Espace)
    trscheme = truncdim(8) & truncerr(1.0e-11)
    env, = leading_boundary(
        env, peps; alg = :sequential, projector_alg = :fullinfinite, 
        trscheme, tol = 1.0e-10, maxiter = 100, verbosity = 3
    )
    @info "Converging CTMRG with χenv = $(χenv)..."
    trscheme = truncdim(χenv) & truncerr(1.0e-11)
    env, = leading_boundary(
        env, peps; alg = :sequential, projector_alg = :fullinfinite, 
        trscheme, tol = 1.0e-10, maxiter = 100, verbosity = 3
    )
    ctm_dir = tps_dir * "ctm-$(χenv)/"
    mkpath(ctm_dir)
    save_CTMRGEnv(ctm_dir, env)

    redirect_stdout()
    redirect_stderr()
    close(logs)
    return nothing
end

folder = ARGS[1]
χenv = parse(Int, ARGS[2])
main_translate(folder, χenv)
