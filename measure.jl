include("imports.jl")
using NPZ
using NaturalSort
using OrderedCollections
using PEPSKitExtras
using PEPSKitExtras: OpstJ

function get_args()
    s = ArgParseSettings(; description = "CTMRG measurement for t-J model")
    @add_arg_table! s begin
        "tps_dir"
        help = "Search pattern for folders containing the iPEPS"
        arg_type = String
        required = true
        "chi"
        help = "Label of the folder containing the CTMRGEnv"
        arg_type = String
        required = true
    end
    args = parse_args(ARGS, s)
    @assert endswith(args["tps_dir"], "/")
    return args
end

function measure_tJ(peps::InfinitePEPS, env::CTMRGEnv, param::AbstractDict)
    model = param["model"]
    psymm, ssymm, slave_fermion = OpstJ.get_symm(model)
    meas = OrderedDict{String, Any}()
    # measure 1-site quantities:
    # doping and magnetization
    nh = OpstJ.gen_siteop("Nh"; model)
    Sx, iSy, Sz = nothing, nothing, nothing
    try
        Sx = OpstJ.gen_siteop("Sx"; model)
    catch end
    try
        iSy = OpstJ.gen_siteop("iSy"; model)
    catch end
    try
        Sz = OpstJ.gen_siteop("Sz"; model)
    catch end
    for (key, op) in zip(("hole", "Sx", "iSy", "Sz"), (nh, Sx, iSy, Sz))
        time0 = time()
        meas["$(key)"] = map(CartesianIndices(peps.A)) do I
            meas_site(I, op, peps, env)
        end
        time1 = time()
        @info @sprintf("Measured %s. Time = %.4f s", key, time1 - time0)
        flush(stdout)
        flush(stderr)
        TensorKit.empty_globalcaches!()
    end
    # measure 1st and 2nd neighbor bond quantities:
    # singlet pairing, hopping, spin correlation, and <n_i n_j>
    num = OpstJ.gen_siteop("Num"; model)
    numnum = num ⊗ num
    hopping = tJ.e_plus_e_min(psymm, ssymm)
    singlet = -real(tJ.singlet_min(psymm, ssymm))
    spincor = real(tJ.S_exchange(psymm, ssymm))
    dirs = ["H", "V"]
    for (key, term) in zip(("hop", "sin", "cor", "nn"), (hopping, singlet, spincor, numnum))
        for dir in dirs
            if (key == "nn") && (dir in ["D1", "D2"])
                continue
            end
            time0 = time()
            meas["$(key)$(dir)"] = collect(
                meas_bond(I, dir, term, peps, env) for I in CartesianIndices(peps.A)
            )
            time1 = time()
            @info @sprintf("Measured %s %s. Time = %.4f s", key, dir, time1 - time0)
            flush(stdout)
            flush(stderr)
            TensorKit.empty_globalcaches!()
        end
    end
    # measure energy
    lat2 = InfiniteSquare(size(peps)...)
    @unpack t, J = param
    t2 = haskey(param, "t2") ? param["t2"] : 0.0
    J2 = haskey(param, "J2") ? param["J2"] : 0.0
    ham = tj2_model(Float64, psymm, ssymm, lat2; t, t2, J, J2, mu = 0.0)
    time0 = time()
    meas["e_site"] = cost_function(peps, env, ham) / prod(size(peps))
    time1 = time()
    @info @sprintf("Measured energy. Time = %.4f s", time1 - time0)
    flush(stdout)
    flush(stderr)
    TensorKit.empty_globalcaches!()
    return meas
end

args = get_args()
search = args["tps_dir"]
tps_dirs = sort((search[1] == '/') ? glob(search[2:end], "/") : glob(search); lt = natural)
χenv = args["chi"]
for tps_dir in tps_dirs
    @info "Measuring $(tps_dir)"
    ctm_dir = tps_dir * "ctm-$(χenv)/"
    model, N1, N2 = "tJ", 1, 1

    if !isfile(tps_dir * "A11.jld2")
        @warn "No iPEPS in $tps_dir."
        continue
    end
    peps = load_iPEPS(tps_dir, N1, N2)
    normalize!.(peps.A, Inf)
    if !isfile(ctm_dir * "c1-11.jld2")
        @warn "No CTMRGEnv in $ctm_dir."
        continue
    end
    env = load_CTMRGEnv(ctm_dir, N1, N2)
    param = Dict(
        "model" => model, "N1" => N1, "N2" => N2, 
        "t" => 1.0, "J" => 0.4, "mu" => 0.0,
    )
    meas = measure_tJ(peps, env, param)
    measfile = ctm_dir * "meas.npz"
    npzwrite(measfile, Dict(meas))
    @info "Measurement result saved at $(measfile)."
end
