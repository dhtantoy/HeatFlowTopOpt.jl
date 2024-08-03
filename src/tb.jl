"""
simulation with on config and save the logs and results to vtk file.
"""
function singlerun(config, vtk_file_prefix, vtk_file_pvd, tb_lg, run_i; debug= false)
    # parameters
    begin 
        # independent parameters
        max_it = config["max_it"]
        N::Int = config["N"]
        ϵ = config["ϵ"]
        ϵ_ratio = config["ϵ_ratio"] 
        save_iter::Int = config["save_iter"]
        save_start::Int = config["save_start"]
        vol = config["vol"]

        # parameters corresponding to motion
        up = config["up"]
        down = config["down"]
        τ₀ = config["τ₀"]
        InitType = config["InitType"]
        InitFile = config["InitFile"]
        InitKey = config["InitKey"]
        motion_tag = config["motion_tag"]
        stable_scheme::UInt = config["stable_scheme"]
        rand_scheme::UInt = config["rand_scheme"]
        correct_rate = config["correct_rate"]
        rand_rate = config["rand_rate"]
        rand_kernel_dim::Int = config["rand_kernel_dim"]
        
        # parameters corresponding to PDE
        β₁ = config["β₁"]
        β₂ = config["β₂"]
        β₃ = config["β₃"]
        α⁻ = config["α⁻"]
        kf = config["kf"]
        ks = config["ks"]
        γ = config["γ"]
        Ts = config["Ts"]
        Re = config["Re"]
        δt = config["δt"]
        δu = config["δu"]
        ud = VectorValue(config["ud⋅n"], 0.)
        g = VectorValue(config["g⋅n"], 0.)
        Td = config["Td"]
        dim = config["dim"]
        L = config["L"]

        τ₀ = iszero(rand_scheme & (RANDOM_WALK | RANDOM_CHANGE)) ? τ₀ : zero(τ₀)
        
        if motion_tag == "conv"
            motion = Conv(Float64, 2, N + 1, τ₀; time= debug ? 1 : 120);
        elseif motion_tag == "gf"
            pdsz = config["pdsz"]
            motion = GaussianFilter(Float64, 2, N + 1, τ₀; pdsz= pdsz)
        else
            error("motion not defined!")
        end

        τ = τ₀
        rand_kernel_dim = (N+1) ÷ rand_kernel_dim
    end

    # ----------------------------------- Host output -----------------------------------
    with_logger(tb_lg) do 
        dict_info = Dict(
            "LOGNAME" => ENV["LOGNAME"],
            "PID" => getpid(),
            "UID" => parse(Int, readchomp(`id -u`)),
            "GID" => parse(Int, readchomp(`id -g`)),
            "WORKER" => myid(),
        )   
        try 
            push!(dict_info, "HOSTNAME" => ENV["HOSTNAME"])
        catch
        end
        @info "host" base=TBText(DataFrame(dict_info)) log_step_increment=0
    end

    # ----------------------------------- model setting -----------------------------------  
    model = CartesianDiscreteModel(repeat([0, L], dim), repeat([N], dim)) |> simplexify

    aux_space, (Ω, _), (dx, dΓs) = initmodel(model, [[5, 6], 7, 8]);
    # ---------------------------------------------------------------------------------------------------------
    
    # init χs and prepare cache for array.
    cache_arr_χ, cache_arr_Gτχ = initcachechis(InitType, aux_space; vol= vol, file= InitFile, key= InitKey);
    cache_arr_χ₂ = zero(cache_arr_χ);
    cache_arr_Gτχ₂ = zero(cache_arr_Gτχ);
    cache_arr_rand_χ = zero(cache_arr_χ);
    cache_rand_kernel = zeros(eltype(cache_arr_χ), (rand_kernel_dim, rand_kernel_dim));

    volₖ = sum(cache_arr_χ) / length(cache_arr_χ);
    M = round(Int, length(cache_arr_χ) * vol);
    χ₀ = copy(cache_arr_χ);
    arr_χ_old = zero(cache_arr_χ);

    # init fe funcs 
    motion_funcs = initfefuncs(aux_space);

    # init spaces
    test_spaces, trial_spaces, assemblers, cache_As, cache_lus, cache_bs, fe_funcs, ad_fe_funcs = initspaces(model, dx, Td, ud)
    
    # allocate cache for computing nodal value and Φ
    # (cache_Φ, cache_rev_Φ, cache_node_val)
    cache_Φ = Matrix{Float64}(undef, N + 1, N + 1);
    cache_Φs = (cache_Φ, zero(cache_Φ), zero(cache_Φ));
    
    debug && @info "run_$(run_i): computing initial energy ..."
    cache_arrs = (cache_arr_χ, cache_arr_χ₂, cache_arr_Gτχ, cache_arr_Gτχ₂);
    params = (α⁻, kf, ks)
    update_motion_funcs!(motion_funcs, cache_arrs, motion, params);

    params = (g, β₁, β₂, β₃, N, Re, δt, δu, γ, Ts, motion.τ[])
    J = pde_solve!(fe_funcs, test_spaces, trial_spaces, cache_As, cache_lus, cache_bs, assemblers,
                params, motion_funcs, dx, dΓs)
    E = +(J...);

    with_logger(tb_lg) do 
        image_χ = TBImage(cache_arr_χ, WH)
        @info "energy" E1= J[1] E2= J[2] E3= J[3] E= E
        @info "domain" χ= image_χ log_step_increment=0
    end

    # allocate cache for iteration and correction
    n = (N+1)^2;
    cache_val_exist = SzVector(Float64, n, 0);
    cache_idx_exist = SzVector(Int, n, 0);
    idx_exist_sort_pre = SzVector(Int, n, 0);
    idx_pick_post = SzVector(Int, n, 0);
    idx_increase = SzVector(Int, n, 0);
    idx_decrease = SzVector(Int, n, 0);
    rand_idx_cache = zeros(Int, n);

    debug && @info "----------------- start iteration ------------------"

    i = 1
    while i <= max_it
        debug && @info "run_$(run_i) iteration $(i): "

        time_out = time()
        copy!(arr_χ_old, cache_arr_χ);

        params = (N, Re, δt, δu, γ, β₃)
        adjoint_pde_solve!(ad_fe_funcs, fe_funcs, test_spaces, trial_spaces, cache_As, cache_lus, cache_bs, assemblers,
                        params, motion_funcs, dx)

        # all pde solved.
        params = (β₁, β₂, β₃, α⁻, Ts, kf, ks, γ)
        Phi!(cache_Φs, params, fe_funcs, ad_fe_funcs, aux_space, motion, cache_arr_Gτχ, cache_arr_Gτχ₂)

        if i >= save_start && mod(i, save_iter) == 0
            Th, uh = fe_funcs
            fe_χ = motion_funcs[1]
            if debug
                ret = Phi_debug(cache_Φs, params, fe_funcs, ad_fe_funcs, aux_space, motion, cache_arr_Gτχ, cache_arr_Gτχ₂)

                # only support for uniform mesh grids.
                V = get_fe_space(fe_χ)
                Φ = FEFunction(V, vec(cache_Φ))
                Thˢ, uhˢ = ad_fe_funcs
                c_fs = [
                    "Th" => Th,
                    "∇(Th)" => ∇(Th),
                    "uh" => uh,
                    "∇⋅uh" => divergence(uh), 
                    "Thˢ" => Thˢ, 
                    "∇(Thˢ)" => ∇(Thˢ),
                    "uhˢ" => uhˢ, 
                    "∇⋅uhˢ" => divergence(uhˢ),
                    "χ" => fe_χ,
                    "Φ" => Φ,
                    "uh⋅uhˢ" => uh⋅uhˢ,
                    "∇(Th)⋅∇(Thˢ)" => ∇(Th)⋅∇(Thˢ),
                    "(Th - Ts) * Thˢ" => (Ts - Th) * Thˢ,
                    "-Re*Thˢ*∇Th" => -Re * Thˢ * ∇(Th),

                    "- β₁/2 * α⁻ * uh⋅uh" => - β₁/2 * α⁻ * uh⋅uh,
                    "- β₁/2 * α⁻ * Gτ(uh⋅uh)" => FEFunction(V, vec(ret[1])),

                    "β₃ * γ * (ks - kf) * (Ts - Th)" => β₃ * γ * (ks - kf) * (Ts - Th),

                    "β₃ * γ * (ks - kf) * Gτ(Ts - Th)" => FEFunction(V, vec(ret[4])),

                    "- α⁻ * (uh⋅uhˢ)" => - α⁻ * (uh⋅uhˢ),
                    "- α⁻ * Gτ(uh⋅uhˢ)" => FEFunction(V, vec(ret[5])),

                    "(kf - ks) * ∇(Th)⋅∇(Thˢ)" => (kf - ks) * ∇(Th)⋅∇(Thˢ),
                    "(kf - ks) * Gτ(∇T⋅∇Tˢ)" => FEFunction(V, vec(ret[6])),

                    "γ * (kf - ks) * (Th - Ts) * Thˢ" => γ * (kf - ks) * (Th - Ts) * Thˢ,
                    "γ * (kf - ks) * Gτ((Th - Ts) * Thˢ)" => FEFunction(V, vec(ret[7])),
                    
                    "Φ̂" => - β₁/2 * α⁻ * uh⋅uh + 
                            β₃ * γ * (ks - kf) * (Ts - Th) + 
                            - α⁻ * (uh⋅uhˢ) + 
                            (kf - ks) * ∇(Th)⋅∇(Thˢ) + 
                            γ * (kf - ks) * (Th - Ts) * Thˢ,
                    ]
            else
                c_fs = ["Th" => Th, "uh" => uh, "χ" => fe_χ]
            end
            vtk_file_pvd[Float64(i)] =  createvtk(Ω, vtk_file_prefix * string(i); cellfields= c_fs)
        end

        # --------------------- pre-process χ ---------------------------------
        ## update on the boundary
        if !iszero(stable_scheme & STABLE_BOUNDARY)
            bd_post_phi!(cache_Φ, cache_arr_Gτχ, down, up)
        end

        if !iszero(rand_scheme & RANDOM_CHANGE)
            m = round(Int, rand_rate * M)
            cache_perm = idx_pick_post.data
            rand_post_phi!(cache_Φ, cache_perm, m, i)
        end
        # ---------------------------------------------------------------------

        get_ordered_idx!(idx_exist_sort_pre, cache_val_exist, cache_idx_exist, cache_arr_χ, cache_Φ);
        iterateχ!(cache_arr_χ, cache_Φ, idx_pick_post, M);

        # --------------------- post-process χ ---------------------------------
        if !iszero(stable_scheme & STABLE_OLD)
            post_chi!(cache_arr_χ, arr_χ_old, 0.5)
        end
        if !iszero(rand_scheme & RANDOM_WALK)
            _vol = M / length(cache_arr_χ)
            random_chi!(cache_arr_rand_χ, cache_rand_kernel, _vol, i);
            post_chi!(cache_arr_χ, cache_arr_rand_χ, rand_rate)
        elseif !iszero(rand_scheme & RANDOM_WINDOW)
            window = cache_arr_rand_χ
            random_chi!(window, cache_rand_kernel, max(rand_rate, 0.1), i) 
            post_window_chi!(cache_arr_χ, arr_χ_old, window)
        end
        # ---------------------------------------------------------------------

        params = (α⁻, kf, ks)
        coeffs = update_motion_funcs!(motion_funcs, cache_arrs, motion, params)

        params = (g, β₁, β₂, β₃, N, Re, δt, δu, γ, Ts, motion.τ[])
        J = pde_solve!(fe_funcs, test_spaces, trial_spaces, cache_As, cache_lus, cache_bs, assemblers,
                    params, motion_funcs, dx, dΓs)
        Ei = +(J...);

        time_out = time() - time_out
        
        time_in = time()
        n_in_iter = 0
        err_flag_in_iter = false
        
        if !iszero(stable_scheme & STABLE_CORRECT)
            Φ_min, Φ_max = extrema(cache_Φ)
            if !iszero(rand_scheme & RANDOM_PROB)
                rand_prob_chi!(cache_Φs, rand_idx_cache, Φ_min, Φ_max)
            end 
            while Ei >= E
                n_in_iter += 1

                n_in_iter > 20 && error("n_in_iter too big!")

                computediffset!(idx_decrease, idx_increase, idx_exist_sort_pre, idx_pick_post, cache_Φ)

                if length(idx_decrease) == 0 && length(idx_increase) == 0 
                    @info "run_$(run_i): correction failed!" 
                    err_flag_in_iter = true
                    break
                end
                # correct_with_phi!(cache_arr_χ, correct_rate, idx_decrease, idx_increase, cache_Φ)
                correct_random!(cache_arr_χ, correct_rate, idx_decrease, idx_increase, rand_idx_cache)
                get_ordered_idx!(idx_pick_post, cache_val_exist, cache_idx_exist, cache_arr_χ, cache_Φ)
                
                params = (α⁻, kf, ks)
                coeffs = update_motion_funcs!(motion_funcs, cache_arrs, motion, params)

                params = (g, β₁, β₂, β₃, N, Re, δt, δu, γ, Ts, motion.τ[])
                J = pde_solve!(fe_funcs, test_spaces, trial_spaces, cache_As, cache_lus, cache_bs, assemblers,
                            params, motion_funcs, dx, dΓs)
                Ei = +(J...)
            end
        end

        # inner iteration failed
        err_flag_in_iter && break

        E = Ei
        time_in = time() - time_in

        volₖ = sum(cache_arr_χ) / length(cache_arr_χ)
        curϵ = norm(cache_arr_χ - arr_χ_old, 2)
        τ = motion.τ[]
        debug && @info "run_$(run_i): J = $(J), E = $E, τ = $(τ), cur_ϵ= $(curϵ), β₂ = $β₂, in_iter= $n_in_iter"
        with_logger(tb_lg) do 
            image_χ = TBImage(cache_arr_χ, WH)
            @info "energy" E1= J[1] E2= J[2] E3= J[3] E= E
            @info "domain" χ= image_χ log_step_increment=0
            @info "parameters" τ= τ  ϵ= curϵ rand_rate=rand_rate log_step_increment=0
            @info "count" in_iter= n_in_iter in_time= time_in out_time= time_out volₖ= volₖ M= M log_step_increment=0
        end

        τ < 1e-8 && break

        curϵ < ϵ && update_tau!(motion, ϵ_ratio)
        
        i += 1
        rand_rate *= 0.99
    end

    vtk_file_pvd[Float64(max_it + 1)] = createvtk(Ω, vtk_file_prefix * string(max_it + 1); 
            cellfields=[
                "Th" => fe_funcs[1], 
                "Thˢ" => ad_fe_funcs[1],
                "uh" => fe_funcs[2],
                "uhˢ" => ad_fe_funcs[2],
                "χ" => motion_funcs[1]
            ]
        )

    return χ₀, cache_arr_χ
end

"""
- parse vec_configs to a vector of configs;
- run the `singlerun` parallelly with one of the configs; 
- make directories: 
    - jld2: save the results in `jld2` format; 
    - tb: save the tensorboard logs; 
    - vtk: save the vtk files; 
    - config.toml: save all the configs in `toml` format; 
    - tensorboard.sh: a script to start tensorboard; 
    - vec_configs.jl: save the `vec_configs` code to reuse directly.
"""
function run_with_configs(vec_configs, comments)
    idx = findfirst(((k, _),) -> k == "max_it", vec_configs)
    max_it = last(vec_configs[idx])

    base_config, appended_config_arr = parse_vec_configs(vec_configs)

    # data path 
    path = joinpath("data", Dates.format(now(), "yyyy-mm-ddTHH_MM_SS"))
    make_path(path, 0o751)

    open(joinpath(path, "vec_configs.jl"), "w") do io
        println(io, vec_configs)
    end
    
    # info
    dict_info = Dict(
        "LOGNAME" => ENV["LOGNAME"],
        "PWD" => ENV["PWD"],
        "PID" => getpid(),
        "NPROCS" => nprocs(),
        "UID" => parse(Int, readchomp(`id -u`)),
        "GID" => parse(Int, readchomp(`id -g`)),
        "COMMENTS" => comments,
        
    )
    try 
        push!(dict_info, "HOSTNAME" => ENV["HOSTNAME"])
    catch 
    end
    try 
        push!(dict_info, "COMMIT" => readchomp(`git rev-parse --short HEAD`))
    catch 
    end


    open(joinpath(path, "config.toml"), "w") do io
        dict_vec_config = Dict(vec_configs)
        dict_vec_config["rand_scheme"] = parse_random_scheme(dict_vec_config["rand_scheme"]) 
        dict_vec_config["stable_scheme"] = parse_stable_scheme(dict_vec_config["stable_scheme"])
        
        out = Dict(
            "info" => dict_info,
            "vec_configs" => dict_vec_config,
        )
        TOML.print(io, out) do el 
            if el isa Symbol
                return String(el)
            end
            return el
        end
    end

    # tensorboard script
    tb_path_prefix = joinpath(dict_info["PWD"], path, "tb")
    make_path(tb_path_prefix, 0o753)

    sh_file_name = joinpath(path, "tensorboard.sh")
    touch(sh_file_name)
    chmod(sh_file_name, 0o755)
    open(sh_file_name, "w") do io
        println(io, "#!/bin/bash")
        println(io, "tensorboard --logdir=./tb --port=\$1 --samples_per_plugin=images=$(max_it+1)")
    end

    jld2_prefix = joinpath(path, "jld2")
    make_path(jld2_prefix, 0o753)

    vtk_prefix = joinpath(path, "vtk")
    make_path(vtk_prefix, 0o753)

    err_prefix = joinpath(path, "errors")
    make_path(err_prefix, 0o753)

    pmap(eachindex(appended_config_arr)) do i
        multi_config = Dict{String, Any}(appended_config_arr[i])
        config = merge(base_config, multi_config)

        tb_file_prefix = joinpath(tb_path_prefix, "run_$i")
        tb_lg = TBLogger(tb_file_prefix, tb_overwrite, min_level=Logging.Info)

        if !isempty(multi_config)
            if haskey(multi_config, "rand_scheme")
                multi_config["rand_scheme"] = parse_random_scheme(multi_config["rand_scheme"]) 
            end
            if haskey(multi_config, "stable_scheme")
                multi_config["stable_scheme"] = parse_stable_scheme(multi_config["stable_scheme"])
            end
            
            write_hparams!(tb_lg, multi_config, ["energy/E"])
        end

        vtk_file_prefix = joinpath(vtk_prefix, "run_$(i)_")
        vtk_file_pvd = createpvd(vtk_file_prefix)
        try
            χ₀, χ = singlerun(config, vtk_file_prefix, vtk_file_pvd, tb_lg, i)
            jld2_file = joinpath(jld2_prefix, "run_$i.jld2")
            jldopen(jld2_file, "w") do _file
                _file["χ₀"] = χ₀
                _file["χ"] = χ
                _file["config"] = config
            end
        catch e
            open(joinpath(err_prefix, "run_$i.err"), "w") do io
                println(io, current_exceptions())
            end
        finally
            close(tb_lg)
            savepvd(vtk_file_pvd)
            nothing
        end
    end
    return nothing
end

function parse_vec_configs(vec_configs::Vector)
    multi_config_pairs = filter(((_, v),) -> isa(v, Vector), vec_configs) 
    single_config_pairs = filter(((_, v),) -> !isa(v, Vector), vec_configs)
        
    base_config = Dict(single_config_pairs)
    broadcasted_config_pairs = map(((k, v),) -> broadcast(Pair, k, v), multi_config_pairs)
    appended_config_arr = Iterators.product(broadcasted_config_pairs...) |> collect

    return base_config, appended_config_arr
end