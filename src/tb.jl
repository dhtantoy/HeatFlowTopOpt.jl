# cache_* denotes the cache for array-form variable with the same type and size.


"""
simulation with on config and save the logs and results to vtk file.
"""
function singlerun(config, vtk_file_prefix, vtk_file_pvd, tb_lg, run_i; debug= false)
    # ----------------------------------- config -----------------------------------
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
        motion_type = config["motion_type"]
        scheme::Unsigned = config["scheme"]
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

        @inline _is_scheme(s::Unsigned) = !iszero( scheme & s )

        if _is_scheme(SCHEME_WALK | SCHEME_CHANGE) || isempty(motion_type)
            τ₀ = zero(τ₀)
            β₂ = 0.
        end
        
        if motion_type == "conv"
            motion = Conv(Float64, 2, N + 1, τ₀; time= debug ? 1 : 120);
        elseif motion_type == "gf"
            pdsz = config["pdsz"]
            motion = GaussianFilter(Float64, 2, N + 1, τ₀; pdsz= pdsz)
        elseif motion_type == ""
            motion = copy!
        else
            error("motion type not defined!")
        end

        τ = τ₀
        h = 1 / N; 
        δt *= h^2; 
        δu *= h^2;
        μ = 1/Re; 
        rand_kernel_dim = (N+1) ÷ rand_kernel_dim
    end

    # ----------------------------------- model setting ----------------------------------- 
    model = CartesianDiscreteModel(repeat([0, L], dim), repeat([N], dim)) |> simplexify;
    aux_space = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity= :H1);
    # -------------------------------------------------------------------------------------

    # ----------------------------------- cache for indicators ---------------------------
    # `fe_*` denotes `FEFunction`-from
    fe_χ = FEFunction(aux_space, ones(num_free_dofs(aux_space)));
    fe_χ₂ = FEFunction(aux_space, zeros(num_free_dofs(aux_space)));
    fe_Gτχ = FEFunction(aux_space, ones(num_free_dofs(aux_space)));
    fe_Gτχ₂ = FEFunction(aux_space, zeros(num_free_dofs(aux_space)));
    
    # changes of `arr_fe_*` will be reflected in `fe_*`. 
    arr_fe_χ = PermArray(fe_χ.free_values, perm, dim);
    arr_fe_χ₂ = PermArray(fe_χ₂.free_values, perm, dim);
    arr_fe_Gτχ = PermArray(fe_Gτχ.free_values, perm, dim);
    arr_fe_Gτχ₂ = PermArray(fe_Gτχ₂.free_values, perm, dim);
    init_chi!(arr_fe_χ, InitType; vol= vol, file= InitFile, key= InitKey);
    n = length(arr_fe_χ)
    debug && display(heatmap(arr_fe_χ))

    arr_rand_χ = zero(arr_fe_χ);
    arr_rand_kernel = zeros(rand_kernel_dim, rand_kernel_dim);
    arr_old_χ = zero(arr_fe_χ);
    arr_init_χ = copy(arr_fe_χ);
    # -------------------------------------------------------------------------------------

    # ----------------------------------- array-form Φ -------------------------
    arr_fe_Φ = PermArray(zeros(num_free_dofs(aux_space)), perm, dim);
    arr_cache_Φ_1 = zero(arr_fe_Φ);
    arr_cache_Φ_2 = zero(arr_fe_Φ);
    arr_old_Φ = zero(arr_fe_Φ);
    # -----------------------------------------------------------------------

    # ----------------------------------- problem-dependent setting -----------------------
    ## heat-flow
    trian = Triangulation(model);
    bdtrain = BoundaryTriangulation(model; tags= 7);
    dx = Measure(trian, 4);
    dσ = Measure(bdtrain, 4);
    α = α⁻ * fe_Gτχ₂; κ = kf * fe_Gτχ + ks * fe_Gτχ₂;

    @inline _a_V((u, p), (v, q)) = ∫(∇(u)⊙∇(v)*μ + u⋅v*α - (∇⋅v)*p + q*(∇⋅u))dx # + ∫(∇(p)⋅∇(q)*δu)dx
    @inline _l_V((v, q)) = ∫( g ⋅ v)dσ
    @inline a_V((uc, ub, p), (vc, vb, q)) = _a_V((uc + ub, p), (vc + vb, q))
    @inline l_V((vc, vb, q)) = _l_V((vc + vb, q))
    VP_test, VP_trial, VP_assem, VP_A, VP_LU, VP_b, Xh, Xhˢ = init_single_space(Val(:StokesMini), trian, a_V, [5, 6], ud)
    uch, ubh, _ = Xh; uchˢ, ubhˢ, _ = Xhˢ;
    uh = uch + ubh; uhˢ = uchˢ + ubhˢ;

    @inline a_T(T, v) = ∫(∇(T) ⋅ ∇(v) * κ + uh⋅∇(T)*v*Re + γ*κ*T*v)dx + ∫((uh⋅∇(T)*Re + γ*κ*T)*(Re*uh⋅∇(v)*δt))dx
    @inline l_T(v) = ∫(γ*κ*Ts*v)*dx + ∫(γ*κ*Ts*Re*uh⋅∇(v)*δt)dx
    T_test, T_trial, T_assem, T_A, T_LU, T_b, Th, Thˢ = init_single_space(Val(:Heat), trian, a_T, [7], Td)

    @inline a_Tˢ(Tˢ, v) = ∫(∇(Tˢ) ⋅ ∇(v) * κ + uh⋅∇(v)*Tˢ*Re + γ*κ*Tˢ*v)dx + ∫((uh⋅∇(Tˢ)*Re - γ*κ*Tˢ)*(Re*uh⋅∇(v))*δt)dx 
    @inline l_Tˢ(v) = ∫(- β₃ * κ *γ * v)dx + ∫(β₃ * κ *γ * (Re*uh⋅∇(v))*δt)dx
    @inline _a_Vˢ((uˢ, pˢ), (v, q)) = ∫(μ*∇(uˢ)⊙∇(v) + uˢ⋅v*α + (∇⋅v)*pˢ - q*(∇⋅uˢ))dx #+ ∫(∇(pˢ)⋅∇(q)*δu)dx
    @inline _l_Vˢ((v, q)) = ∫(-(∇(Th))⋅v*Re*Thˢ)dx #+ ∫(-(∇(Th))⋅ ∇(q) *Re*Thˢ * δu)dx
    @inline a_Vˢ((uc, ub, p), (vc, vb, q)) = _a_Vˢ((uc + ub, p), (vc + vb, q))
    @inline l_Vˢ((vc, vb, q)) = _l_Vˢ((vc + vb, q))

    cell_fields = ["Th" => Th, "uh" => uh, "χ" => fe_χ]
    # -------------------------------------------------------------------------------------

    # ----------------------------------- initial quantities -----------------------------------
    volₖ = sum(arr_fe_χ) / n;
    M = round(Int, n * vol);
    smooth_funcs!(arr_fe_χ, arr_fe_χ₂, arr_fe_Gτχ, arr_fe_Gτχ₂, motion) 
    pde_solve!(Xh, a_V, l_V, VP_test, VP_trial, VP_A, VP_LU, VP_b, VP_assem)
    pde_solve!(Th, a_T, l_T, T_test, T_trial, T_A, T_LU, T_b, T_assem)

    Ju = β₁/2 * sum( _a_V((uh, 0), (uh, 0)) )
    Jγ = β₂ * sqrt(π/τ) * sum( ∫(fe_χ * fe_Gτχ₂)dx )
    @check_tau(Jγ)
    Jt = β₃* sum( ∫((Th - Ts)*κ*γ)dx )
    E = Ju + Jγ + Jt
    # -------------------------------------------------------------------------------------

    # ----------------------------------- log output -----------------------------------
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
        catch; finally
            image_χ = TBImage(arr_fe_χ, WH)
            @info "domain" χ= image_χ log_step_increment=0
            @info "host" base=TBText(DataFrame(dict_info)) log_step_increment=0
            @info "energy" Ju= Ju Jγ= Jγ Jt= Jt J= J
        end
    end
    # -------------------------------------------------------------------------------------
    
    # ----------------------------------- prepare for correction -----------------------
    sz_val = SzVector(Float64, n, 0);
    sz_idx = SzVector(Int, n, 0);
    idx_A = SzVector(Int, n, 0);
    idx_B = SzVector(Int, n, 0);
    sorted_idx_inc = SzVector(Int, n, 0);
    sorted_idx_dec = SzVector(Int, n, 0);
    vec_idx = sz_idx.data
    # -------------------------------------------------------------------------------------
    
    # ----------------------------------- iterations -----------------------------------
    i = 1
    while i <= max_it
        debug && @info "run_$(run_i) iteration $(i): "
        time_out = time()

        copy!(arr_old_χ, arr_fe_χ);
        copy!(arr_old_Φ, arr_fe_Φ);

        # ---- solve adjoint pde
        pde_solve!(Thˢ, a_Tˢ, l_Tˢ, T_test, T_trial, T_A, T_LU, T_b, T_assem)
        pde_solve!(Xhˢ, a_Vˢ, l_Vˢ, VP_test, VP_trial, VP_A, VP_LU, VP_b, VP_assem)

        # ---- now all pde solved, then compute Φ
        ## base gradient of energy
        fe_Φ = -α⁻ * (β₁/2*uh⋅uh + 2*uh⋅uhˢ) + (kf - ks)*(∇(Th)⋅∇(Thˢ)) + γ*(ks - kf)*((Ts - Th)*(Thˢ + β₃))
        _compute_node_value!(arr_fe_Φ, fe_Φ, trian)
        motion(arr_cache_Φ_1, arr_fe_Φ)
        ## perimeter of interface
        _r = β₂ * sqrt(π / τ); @check_tau(_r);
        @. arr_cache_Φ_1 += _r * (arr_fe_Gτχ₂ - arr_fe_Gτχ)
        ## post-processing for symmetry
        symmetry!(arr_fe_Φ, arr_cache_Φ_1; dims= 2)

        # ---- saving data
        if i >= save_start && mod(i, save_iter) == 0
            if debug
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
                    "fe_Φ" => fe_Φ,
                    "uh⋅uhˢ" => uh⋅uhˢ,
                    "∇(Th)⋅∇(Thˢ)" => ∇(Th)⋅∇(Thˢ),
                    "(Th - Ts) * Thˢ" => (Ts - Th) * Thˢ,
                    "-Re*Thˢ*∇Th" => -Re * Thˢ * ∇(Th),
                    "- β₁/2 * α⁻ * uh⋅uh" => - β₁/2 * α⁻ * uh⋅uh,
                    "β₃ * γ * (ks - kf) * (Ts - Th)" => β₃ * γ * (ks - kf) * (Ts - Th),
                    "- α⁻ * (uh⋅uhˢ)" => - α⁻ * (uh⋅uhˢ),
                    "(kf - ks) * ∇(Th)⋅∇(Thˢ)" => (kf - ks) * ∇(Th)⋅∇(Thˢ),
                    "γ * (kf - ks) * (Th - Ts) * Thˢ" => γ * (kf - ks) * (Th - Ts) * Thˢ,
                    ]
            else
                c_fs = cell_fields
            end
            vtk_file_pvd[Float64(i)] =  createvtk(trian, vtk_file_prefix * string(i); cellfields= c_fs)
        end

        # ---- pre-process χ
        ## stablization with Φ_k when i >= 2, note that when i = 1, arr_Φ_old = 0
        _is_scheme(SCHEME_OLD_PHI) && post_interpolate!(arr_fe_Φ, arr_old_Φ, 0.5)

        Φ_min, Φ_max = extrema(arr_fe_Φ)

        ## update on the boundary
        _is_scheme(SCHEME_BOUNDARY) && post_phi!(arr_fe_Φ, arr_fe_Gτχ, Φ_max, Φ_min, down, up)

        ## random change
        _is_scheme(SCHEME_CHANGE) && rand_post_phi!(arr_fe_Φ, vec_idx, Φ_max, round(Int, rand_rate * M), i)
        
        ## random correction
        if _is_scheme(SCHEME_PROB_CORRECT)
            P = arr_cache_Φ_1
            weight = arr_cache_Φ_2
            phi_to_prob!(P, arr_fe_Φ, Φ_max)
            prob_to_weight!(weight, P, vec_idx)
        elseif _is_scheme(SCHEME_RAND_CORRECT)
            randperm!(Random.seed!(i), vec_idx)
            weight = arr_cache_Φ_2
            weight[vec_idx] = 1. :length(weight)
        elseif _is_scheme(SCHEME_CORRECT_REV)
            weight = -arr_fe_Φ
        else
            weight = arr_fe_Φ
        end

        # ---- iteration
        ## get selected indices of χ_k in ascending order under weight
        get_sorted_idx!(idx_A, sz_val, sz_idx, arr_fe_χ, weight);
        ## get χ_{k + 1} and seleted indices in ascending order of it.
        ## note that `idx_B` for `SCHEME_CORRECT` is updated correctly too.
        
        iterateχ!(arr_fe_χ, idx_B, arr_fe_Φ, M);

        ## get selected indices of χ_{k + 1} in ascending order under weight
        ## for correction-scheme except `SCHEME_CORRECT`.
        _is_scheme(remove_bitmode(SCHEME_ALL_CORRECT, SCHEME_CORRECT)) && get_sorted_idx!(idx_B, sz_val, sz_idx, arr_fe_χ, weight)

        # ---- post-process χ 
        _is_scheme(SCHEME_OLD) && post_interpolate!(arr_fe_χ, arr_old_χ, 0.5)

        if _is_scheme(SCHEME_WALK)
            _vol = M / n
            random_window!(arr_rand_χ, arr_rand_kernel, _vol, i);
            post_interpolate!(arr_fe_χ, arr_rand_χ, rand_rate)
        end
        if _is_scheme(SCHEME_WINDOW)
            window = arr_rand_χ
            random_window!(window, arr_rand_kernel, max(rand_rate, 0.1), i) 
            post_interpolate!(arr_fe_χ, arr_old_χ, window)
        end
        
        # ---- compute energy
        smooth_funcs!(arr_fe_χ, arr_fe_χ₂, arr_fe_Gτχ, arr_fe_Gτχ₂, motion)
        ## solve pde with χ_{k+1} 
        pde_solve!(Xh, a_V, l_V, VP_test, VP_trial, VP_A, VP_LU, VP_b, VP_assem)
        pde_solve!(Th, a_T, l_T, T_test, T_trial, T_A, T_LU, T_b, T_assem)
        ## energy
        Ju = β₁/2 * sum( _a_V((uh, 0), (uh, 0)) )
        Jγ = β₂ * sqrt(π/τ) * sum( ∫(fe_χ * fe_Gτχ₂)dx )
        @check_tau(Jγ)
        Jt = β₃* sum( ∫((Th - Ts)*κ*γ)dx )
        Ji = Ju + Jγ + Jt

        time_out = time() - time_out
        

        debug && display(heatmap(arr_fe_χ))
        # ---- prediction correction
        time_in = time()
        n_in_iter = 0
        flag_in_iter_stop = false
        if _is_scheme(SCHEME_ALL_CORRECT)
            Ji >= J && computediffset!(sorted_idx_dec, sorted_idx_inc, idx_A, idx_B, weight)
            while Ji >= J
                n_in_iter += 1
                if length(sorted_idx_dec) == 0 && length(sorted_idx_inc) == 0 
                    flag_in_iter_stop = true
                    break
                end

                nonsym_correct!(arr_fe_χ, sorted_idx_dec, sorted_idx_inc, correct_rate)

                debug && display(heatmap(arr_fe_χ))

                # ---- compute energy
                smooth_funcs!(arr_fe_χ, arr_fe_χ₂, arr_fe_Gτχ, arr_fe_Gτχ₂, motion)
                ## solve pde with χ_{k+1} 
                pde_solve!(Xh, a_V, l_V, VP_test, VP_trial, VP_A, VP_LU, VP_b, VP_assem)
                pde_solve!(Th, a_T, l_T, T_test, T_trial, T_A, T_LU, T_b, T_assem)
                ## energy
                Ju = β₁/2 * sum( _a_V((uh, 0), (uh, 0)) )
                Jγ = β₂ * sqrt(π/τ) * sum( ∫(fe_χ * fe_Gτχ₂)dx )
                @check_tau(Jγ)
                Jt = β₃* sum( ∫((Th - Ts)*κ*γ)dx )
                Ji = Ju + Jγ + Jt
            end
        end

        # ---- inner iteration failed
        if flag_in_iter_stop
            @warn "inner iteration failed!"
            break
        end

        # ---- log output
        time_in = n_in_iter == 0 ? 0. : time() - time_in
        volₖ = sum(arr_fe_χ) / n
        curϵ = norm(arr_fe_χ - arr_old_χ, 2)

        τ = get_tau(motion)
        debug && @info "run_$(run_i): E = $Ji, τ = $(τ), cur_ϵ= $(curϵ), β₂ = $β₂, in_iter= $n_in_iter"
        with_logger(tb_lg) do 
            image_χ = TBImage(arr_fe_χ, WH)
            @info "energy" Ju= Ju Jγ= Jγ Jt= Jt J= J
            @info "domain" χ= image_χ log_step_increment=0
            @info "parameters" τ= τ  ϵ= curϵ rand_rate=rand_rate log_step_increment=0
            @info "count" in_iter= n_in_iter in_time= time_in out_time= time_out volₖ= volₖ M= M log_step_increment=0
        end

        # ---- update quantities
        curϵ < ϵ && update_tau!(motion, ϵ_ratio)
        if τ < 1e-8 
            @warn "τ is less than 1e-8, now break."
            break
        end
        J = Ji
        i += 1
        rand_rate *= 0.99
    end

    vtk_file_pvd[Float64(max_it + 1)] = createvtk(trian, vtk_file_prefix * string(max_it + 1); cellfields= cell_fields)

    return arr_init_χ, arr_fe_χ
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
    path = joinpath("data", dformat(now(), "yyyy-mm-ddTHH_MM_SS"))
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
        dict_vec_config["scheme"] = scheme_to_str(dict_vec_config["scheme"]) 

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
            if haskey(multi_config, "scheme")
                multi_config["scheme"] = scheme_to_str(multi_config["scheme"]) 
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