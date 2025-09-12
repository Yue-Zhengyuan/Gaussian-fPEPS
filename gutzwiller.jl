include("imports.jl")
cd(@__DIR__)

"""
Perform Gutzwiller projection and measure doping after projection
"""
function sub_gutzwiller(peps0::InfinitePEPS, env0::CTMRGEnv, z::Float64)
    peps = BCS.gutzwiller_project(z, peps0)
    # use FixedSpaceTruncation for CTMRG
    env, = leading_boundary(
        env0, peps; alg = :sequential, projector_alg = :fullinfinite, 
        tol = 1.0e-10, maxiter = 100
    )
    lattice = collect(space(t, 1) for t in peps.A)
    O = LocalOperator(lattice, ((1, 1),) => tJ.h_num(Trivial, Trivial))
    δ = real(expectation_value(peps, O, env))
    return δ, peps, env
end

"""
Main program to perform bisection search of Gutzwiller fugacity `z`
until the change of doping after projection is smaller than `atol`
"""
function main_gutzwiller(
        peps0::InfinitePEPS, env0::CTMRGEnv;
        δ0::Float64, zmax::Float64, atol::Float64,
        zin::Union{Float64, Nothing}
    )
    @info "Exact target doping δ0 = $(δ0)."

    lattice = collect(space(t, 1) for t in peps0.A)
    O = LocalOperator(lattice, ((1, 1),) => hub.e_num(Trivial, Trivial))
    δ0′ = 1 - real(expectation_value(peps0, O, env0))
    @info "PEPS doping before projection δ0′ = $(δ0′)."

    if !isnothing(zin)
        δ, peps, env = sub_gutzwiller(peps0, env0, zin)
        @info "Using provided z = $(zin); doping after projection δ = $δ."
        return δ, peps, env
    end

    zmin = 2 * δ0 / (1 + δ0)
    # check z0, zmax
    δmin, peps, env = sub_gutzwiller(peps0, env0, zmin)
    @assert δmin < δ0
    @info "Doping at zmin = $(δmin) < δ0."
    δmax, peps, env = sub_gutzwiller(peps0, env0, zmax)
    @assert δmax > δ0
    @info "Doping at zmax = $(δmax) > δ0."

    # bisection search of z ∈ (zmin, zmax)
    count, diff = 0, Inf
    z0, z1 = zmin, zmax
    while abs(diff) > atol
        count += 1
        z = (z0 + z1) / 2
        δ, peps, env = sub_gutzwiller(peps0, (diff < 0.005) ? env : env0, z)
        @info "Iter $(count): z = $z, δ = $δ."
        diff = δ - δ0
        (δ > δ0) ? (z1 = z) : (z0 = z)
    end

    # return midpoint of final interval
    z = (z0 + z1) / 2
    δ, peps, env = sub_gutzwiller(peps0, env, z)
    @info "Converged: z = $z, δ = $δ."

    return δ, peps, env
end

function main(
        all_dir::AbstractString, χenv::AbstractString; 
        zmax::Float64, atol::Float64, zin::Union{Float64, Nothing}
    )
    # load the target doping input when optimizing the PEPS cormat
    data = h5open(all_dir * "data.h5", "r")
    δ0 = data["model/doping"][]
    # load PEPS and CTMRGEnv before projection
    tps_dir = all_dir * "noproj/"
    ctm_dir = tps_dir * "ctm-$(χenv)/"
    Nr, Nc = 1, 1
    peps0 = load_iPEPS(tps_dir, Nr, Nc)
    env0 = load_CTMRGEnv(ctm_dir, Nr, Nc)
    # create log files
    tps_dir = replace(tps_dir, "noproj" => "proj-$(χenv)")
    ctm_dir = replace(ctm_dir, "noproj" => "proj-$(χenv)")
    mkpath(tps_dir)
    mkpath(ctm_dir)
    logs = open(tps_dir * "gutzwiller_log.txt", "w")
    redirect_stdout(logs)
    redirect_stderr(logs)
    # projection
    δ, peps, env = main_gutzwiller(peps0, env0; δ0, zmax, atol, zin)
    # save projected PEPS and CTMRGEnv
    save_iPEPS(tps_dir, peps)
    save_CTMRGEnv(ctm_dir, env)
    redirect_stdout()
    redirect_stderr()
    close(logs)
    return nothing
end

function get_args()
    s = ArgParseSettings(; description = "Gutzwiller projection of BCS Gaussian fPEPS.")
    @add_arg_table! s begin
        "all_dir"
        help = "Path to the folder to load and save PEPS."
        arg_type = String
        required = true
        "chi"
        help = "CTMRGEnv bond dimension to be used"
        arg_type = String
        required = true
        "--zmax"
        help = "Maximal Gutzwiller fugacity z to be tested."
        arg_type = Float64
        default = 0.6
        "--atol"
        help = "Error atol of doping after projection"
        arg_type = Float64
        default = 1.0e-8
        "--zin"
        help = "Directly provide the values of Gutzwiller fugacity z"
        arg_type = Float64
    end
    args = parse_args(ARGS, s)
    return args
end

args = get_args()
@unpack all_dir, chi, zmax, zin, atol = args
main(all_dir, chi; zmax, zin, atol)
