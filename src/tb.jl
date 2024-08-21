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
        β = config["β"]
        k1 = config["k1"]
        k2 = config["k2"]
        q1 = config["q1"]
        q2 = config["q2"]
        Td = config["Td"]
        
        dim = config["dim"]

        @inline _is_scheme(s::Unsigned) = !iszero( scheme & s )

        if _is_scheme(SCHEME_WALK | SCHEME_CHANGE)
            τ₀ = zero(τ₀)
            β = 0.
        end
        
        if motion_type == "conv"
            motion = Conv(Float64, 2, N + 1, τ₀; time= debug ? 1 : 120);
        elseif motion_type == "gf"
            pdsz = config["pdsz"]
            motion = GaussianFilter(Float64, 2, N + 1, τ₀; pdsz= pdsz)
        else
            error("motion type not defined!")
        end

        τ = τ₀
        rand_kernel_dim = (N+1) ÷ rand_kernel_dim
    end

    # ----------------------------------- model setting ----------------------------------- 
    diri_tag = "dirichlet"
    model = CartesianDiscreteModel(repeat([0, 1.], dim), repeat([N], dim)) |> simplexify;
    add_tag_by_filter!(model, x -> x[1] == 0. && 0.45 <= x[2] <= 0.55, diri_tag)
    aux_space = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity= :H1);
    # -------------------------------------------------------------------------------------

    # ----------------------------------- cache for indicators ---------------------------
    fe_χ = FEFunction(aux_space, zeros(Float64, num_free_dofs(aux_space)));
    fe_χ₂ = FEFunction(aux_space, zeros(Float64, num_free_dofs(aux_space)));
    fe_Gτχ = FEFunction(aux_space, ones(Float64, num_free_dofs(aux_space)));
    fe_Gτχ₂ = FEFunction(aux_space, ones(Float64, num_free_dofs(aux_space)));
    
    # changes of `cache_fe_arr_*` alter the values of `fe_*`.
    fe_arr_χ = init_chi!(fe_χ, InitType, aux_space; vol= vol, file= InitFile, key= InitKey);
    fe_arr_χ₂ = reshape(fe_χ₂.free_values, size(fe_arr_χ));
    fe_arr_Gτχ = reshape(fe_Gτχ.free_values, size(fe_arr_χ));
    fe_arr_Gτχ₂ = reshape(fe_Gτχ₂.free_values, size(fe_arr_χ));

    arr_rand_χ = zero(fe_arr_χ);
    arr_rand_kernel = zeros(eltype(fe_arr_χ), (rand_kernel_dim, rand_kernel_dim));
    arr_χ_old = zero(fe_arr_χ);
    arr_χ₀ = copy(fe_arr_χ);
    # -------------------------------------------------------------------------------------

    # ----------------------------------- array-form Φ -------------------------
    arr_Φ = Matrix{Float64}(undef, N + 1, N + 1);
    cache_arr_Φ_1 = zero(arr_Φ);
    cache_arr_Φ_2 = zero(arr_Φ);
    # -----------------------------------------------------------------------

    # ----------------------------------- problem-dependent setting -----------------------
    ## heat-flow
    trian = Triangulation(model);
    dx = Measure(trian, 4);
    κ = k1 * fe_Gτχ + k2 * fe_Gτχ₂; q = q1 * fe_Gτχ + q2 * fe_Gτχ₂;
    a(u, v) = ∫(∇(u) ⋅ ∇(v) * κ )dx
    l(v) = ∫(q*v)dx
    test, trial, assem, A, LU, b, Th, _ = init_single_space(Val(:Heat), trian, a, diri_tag, Td)

    cell_fields = ["Th" => Th, "χ" => fe_χ]
    # -------------------------------------------------------------------------------------

    # ----------------------------------- initial quantities -----------------------------------
    debug && @info "run_$(run_i): computing initial energy ..."

    volₖ = sum(fe_arr_χ) / length(fe_arr_χ);
    M = round(Int, length(fe_arr_χ) * vol);
    update_domain_funcs!(fe_arr_χ, fe_arr_χ₂, fe_arr_Gτχ, fe_arr_Gτχ₂, motion) 
    pde_solve!(Th, a, l, test, trial, A, LU, b, assem)

    Jγ = β * sqrt(π/τ) * sum( ∫(fe_χ * fe_Gτχ₂)dx )
    @check_tau(Jγ)
    Jt = sum( l(Th) )
    E = Jγ + Jt
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
            image_χ = TBImage(fe_arr_χ, WH)
            @info "energy" Jγ= Jγ Jt= Jt E= E
            @info "domain" χ= image_χ log_step_increment=0
            @info "host" base=TBText(DataFrame(dict_info)) log_step_increment=0
        end
    end
    # -------------------------------------------------------------------------------------
    
    # ----------------------------------- prepare for correction -----------------------
    n = (N+1)^2;
    cache_sz_val = SzVector(Float64, n, 0);
    cache_sz_idx = SzVector(Int, n, 0);
    idx_A = SzVector(Int, n, 0);
    idx_B = SzVector(Int, n, 0);
    sorted_idx_inc = SzVector(Int, n, 0);
    sorted_idx_dec = SzVector(Int, n, 0);
    cache_arr_idx = cache_sz_idx.data
    # -------------------------------------------------------------------------------------
    
    # ----------------------------------- iterations -----------------------------------
    i = 1
    while i <= max_it
        debug && @info "run_$(run_i) iteration $(i): "
        time_out = time()

        copy!(arr_χ_old, fe_arr_χ);

        # ---- solve adjoint pde
        # Thˢ = - Th
        # pde_solve!(Thˢ, a_Tˢ, l_Tˢ, test, trial, A, LU, b, assem)

        # ---- now all pde solved, then compute Φ
        ## base gradient of energy
        fe_Φ = 2*(q1-q2)*Th - (k1-k2)*(∇(Th)⋅∇(Th))
        _compute_node_value!(cache_arr_Φ_1, fe_Φ, trian)
        motion(arr_Φ, cache_arr_Φ_1)
        ## perimeter of interface
        _r = β * sqrt(π / τ); @check_tau(_r);
        @turbo @. arr_Φ += _r * (fe_arr_Gτχ₂ - fe_arr_Gτχ)
        ## post-processing for symmetry
        copy!(cache_arr_Φ_1, arr_Φ)
        reverse!(cache_arr_Φ_1, dims= 2)
        @turbo @. arr_Φ = (arr_Φ + cache_arr_Φ_1) / 2

        # ---- saving data
        if i >= save_start && mod(i, save_iter) == 0
            vtk_file_pvd[Float64(i)] =  createvtk(trian, vtk_file_prefix * string(i); cellfields= cell_fields)
        end

        # ---- pre-process χ
        Φ_min, Φ_max = extrema(arr_Φ)
        ## update on the boundary
        _is_scheme(SCHEME_BOUNDARY) && post_phi!(arr_Φ, fe_arr_Gτχ, Φ_max, Φ_min, down, up)

        ## random change
        _is_scheme(SCHEME_CHANGE) && rand_post_phi!(arr_Φ, cache_arr_idx, Φ_max, round(Int, rand_rate * M), i)
        
        ## random correction
        if _is_scheme(SCHEME_R_CORRECT)
            P = cache_arr_Φ_1
            weight = cache_arr_Φ_2
            phi_to_prob!(P, arr_Φ, Φ_max)
            prob_to_weight!(weight, P, cache_arr_idx)
        else
            weight = arr_Φ
        end

        # ---- iteration
        ## get selected indices of χ_k in ascending order under weight
        get_sorted_idx!(idx_A, cache_sz_val, cache_sz_idx, fe_arr_χ, weight);
        ## get χ_{k + 1} and seleted indices in ascending order of it.
        iterateχ!(fe_arr_χ, idx_B, arr_Φ, M);
        _is_scheme(SCHEME_R_CORRECT) && get_sorted_idx!(idx_B, cache_sz_val, cache_sz_idx, fe_arr_χ, weight)

        # ---- post-process χ 
        _is_scheme(SCHEME_OLD) && post_chi!(fe_arr_χ, arr_χ_old, 0.5)

        if _is_scheme(SCHEME_WALK)
            _vol = M / length(fe_arr_χ)
            random_window!(arr_rand_χ, arr_rand_kernel, _vol, i);
            post_chi!(fe_arr_χ, arr_rand_χ, rand_rate)
        end
        if _is_scheme(SCHEME_WINDOW)
            window = arr_rand_χ
            random_window!(window, arr_rand_kernel, max(rand_rate, 0.1), i) 
            post_chi!(fe_arr_χ, arr_χ_old, window)
        end
        
        # ---- compute energy
        update_domain_funcs!(fe_arr_χ, fe_arr_χ₂, fe_arr_Gτχ, fe_arr_Gτχ₂, motion)
        ## solve pde with χ_{k+1} 
        pde_solve!(Th, a, l, test, trial, A, LU, b, assem)
        ## energy
        Jγ = β * sqrt(π/τ) * sum( ∫(fe_χ * fe_Gτχ₂)dx )
        @check_tau(Jγ)
        Jt = sum( l(Th) )
        Ei = Jγ + Jt

        time_out = time() - time_out
        
        # ---- prediction correction
        time_in = time()
        n_in_iter = 0
        flag_in_iter_stop = false
        if _is_scheme(SCHEME_CORRECT | SCHEME_R_CORRECT)
            Ei >= E && computediffset!(sorted_idx_dec, sorted_idx_inc, idx_A, idx_B, weight)
            while Ei >= E
                n_in_iter += 1
                if length(sorted_idx_dec) == 0 && length(sorted_idx_inc) == 0 
                    flag_in_iter_stop = true
                    break
                end

                nonsym_correct!(fe_arr_χ, sorted_idx_dec, sorted_idx_inc, correct_rate)

                # ---- compute energy
                update_domain_funcs!(fe_arr_χ, fe_arr_χ₂, fe_arr_Gτχ, fe_arr_Gτχ₂, motion)
                ## solve pde with χ_{k+1} 
                pde_solve!(Th, a, l, test, trial, A, LU, b, assem)
                ## energy
                Jγ = β * sqrt(π/τ) * sum( ∫(fe_χ * fe_Gτχ₂)dx )
                @check_tau(Jγ)
                Jt = sum( l(Th) )
                Ei = Jγ + Jt
            end
        end

        # ---- inner iteration failed
        if flag_in_iter_stop
            @warn "inner iteration failed!"
            break
        end

        # ---- log output
        time_in = n_in_iter == 0 ? 0. : time() - time_in
        volₖ = sum(fe_arr_χ) / length(fe_arr_χ)
        curϵ = norm(fe_arr_χ - arr_χ_old, 2)

        τ = motion.τ[]
        debug && @info "run_$(run_i): E = $Ei, τ = $(τ), cur_ϵ= $(curϵ), β = $β, in_iter= $n_in_iter"
        with_logger(tb_lg) do 
            image_χ = TBImage(fe_arr_χ, WH)
            @info "energy" Jγ= Jγ Jt= Jt E= E
            @info "domain" χ= image_χ log_step_increment=0
            @info "parameters" τ= τ  ϵ= curϵ rand_rate=rand_rate log_step_increment=0
            @info "count" in_iter= n_in_iter in_time= time_in out_time= time_out volₖ= volₖ M= M log_step_increment=0
        end

        # ---- update quantities
        curϵ < ϵ && update_tau!(motion, ϵ_ratio)
        E = Ei
        i += 1
        rand_rate *= 0.99
    end

    vtk_file_pvd[Float64(max_it + 1)] = createvtk(trian, vtk_file_prefix * string(max_it + 1); cellfields= cell_fields)

    return arr_χ₀, fe_arr_χ
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