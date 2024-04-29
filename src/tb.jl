
function singlerun(config, vtk_prefix, tb_lg)
    # independent parameters
    max_it = config["max_it"]
    N::Int = config["N"]
    ϵ = config["ϵ"]
    ϵ_ratio = config["ϵ_ratio"] 
    correct_ratio = config["correct_ratio"]
    save_iter::Int = config["save_iter"]
    vol = config["vol"]

    # parameters corresponding to motion
    up = config["up"]
    down = config["down"]
    τ₀ = config["τ₀"]
    motion_tag = config["motion_tag"] |> Symbol
    InitType = config["InitType"] |> Symbol
    
    # parameters corresponding to PDE
    β₁ = config["β₁"]
    β₂ = config["β₂"]
    β₃ = config["β₃"]
    α⁻ = config["α⁻"]
    Re = config["Re"]
    δt = config["δt"]
    ud = VectorValue(config["ud⋅n"], 0.)
    g = VectorValue(config["g⋅n"], 0.)
    Td = config["Td"]
    dim = config["dim"]
    L = config["L"]
    
    if motion_tag == :conv
        motion = Conv(Float64, 2, N + 1, τ₀; time= 120);
    else
        error("motion not defined!")
    end

    cache_Gτχ = similar(motion.out)

    m, lab, cache_χ, aux_space, (Ω, _...), (dx, dΓin, _...) = initmodel(Val(InitType), N, dim, L);
    χ₀ = copy(cache_χ)

    volₖ = sum(cache_χ) / length(cache_χ);

    # (T_test, T_trial, X, Y), 
    # (T_assem, V_assem, cache_T_b, cache_T_A, cache_V_b, cache_V_A),
    # (cache_Th, cache_Thˢ, cache_uh, cache_uhˢ)
    spaces, cache_fem, cache_fefunc = initspace(m, lab, dx, ud, Td);
    
    # allocate cache for computing nodal value and Φ
    # (cache_Φ, cache_rev_Φ, cache_node_val)
    cache_Φ = Matrix{Float64}(undef, N + 1, N + 1);
    cache = (cache_Φ, similar(cache_Φ), similar(cache_Φ));
    
    # allocate cache for χ_old
    χ_old = similar(cache_χ);

    # fem params 
    pde_params = (g, β₁, β₂, β₃, N, Re, α⁻, δt)
    
    @info "computing initial energy ..."
    # return J
    cache_coeff = _coeff_cache(cache_χ, motion, aux_space, α⁻)
    J = pde_solve!(cache_fem, cache_fefunc, cache_coeff, pde_params, spaces, dx, dΓin, motion)
    E = +(J...);
    with_logger(tb_lg) do 
        image_χ = TBImage(cache_χ, WH)
        @info "energy" E1= J[1] E2= J[2] E3= J[3] E= E 
        @info "domain" χ= image_χ log_step_increment=0
    end

    τ = τ₀

    # allocate cache for iteration and correction
    n = (N+1)^2;
    cache_val_exist = SzVector(Float64, n, 0);
    cache_idx_exist = SzVector(Float64, n, 0);
    idx_exist_sort_pre = SzVector(Int, n, 0);
    idx_pick_post = SzVector(Int, n, 0);
    idx_increase = SzVector(Int, n, 0);
    idx_decrease = SzVector(Int, n, 0);

    @info "----------------- start iteration ------------------"

    i = 1
    while i <= max_it
        @info "iteration $(i): "

        time_out = time()
        copy!(χ_old, cache_χ);
        copy!(cache_Gτχ, motion.out);

        adjoint_pde_solve!(cache_fem, cache_fefunc, cache_coeff, pde_params, spaces, dx)
        Phi!(cache, pde_params, cache_fefunc, Ω, motion)
        post_phi!(cache_Φ, cache_Gτχ, down, up)

        M = count(<=(0), cache_Φ)

        get_ordered_idx!(idx_exist_sort_pre, cache_val_exist, cache_idx_exist, cache_χ, cache_Φ);

        iterateχ!(cache_χ, cache_Φ, idx_pick_post, M);

        # return Th, Thˢ, uh, uhˢ, nothing/J
        cache_coeff = _coeff_cache(cache_χ, motion, aux_space, α⁻)
        J = pde_solve!(cache_fem, cache_fefunc, cache_coeff, pde_params, spaces, dx, dΓin, motion)
        Ei = +(J...)

        time_out = time() - time_out
        
        # correction
        time_in = time()
        n_in_iter = 0
        err_flag_in_iter = false
        while Ei >= E
            n_in_iter += 1

            n_in_iter > 20 && error("n_in_iter too big!")

            computeAB!(idx_decrease, idx_increase, idx_exist_sort_pre, idx_pick_post, cache_Φ)

            if length(idx_decrease) == 0 && length(idx_increase) == 0 
                @info "correction failed! now restart with a smaller τ." 
                err_flag_in_iter = true
                break
            end
            correct!(cache_χ, correct_ratio, idx_decrease, idx_increase, cache_Φ)

            get_ordered_idx!(idx_pick_post, cache_val_exist, cache_idx_exist, cache_χ, cache_Φ)
            
            cache_coeff = _coeff_cache(cache_χ, motion, aux_space, α⁻)
            J = pde_solve!(cache_fem, cache_fefunc, cache_coeff, pde_params, spaces, dx, dΓin, motion)
            Ei = +(J...)
        end

        τ = motion.τ[]

        if τ < 1e-8 
            @info "τ < 1e-8 and break iteration."
            break
        end
        
        # restar with a smaller τ
        if err_flag_in_iter
            update_tau!(motion, ϵ_ratio);
            # restore cache data
            copy!(cache_χ, χ_old);
            copy!(motion.out, cache_Gτχ)
            continue
        end

        E = Ei
        time_in = time() - time_in

        volₖ = sum(cache_χ) / length(cache_χ)
        curϵ = norm(cache_χ - χ_old, 2)
        @info "J = $(J), E = $E, τ = $(τ), cur_ϵ= $(curϵ), β₂ = $β₂, in_iter= $n_in_iter"
        if mod(i, save_iter) == 0 
            writevtk(Ω, joinpath(vtk_prefix, string(i)); 
                cellfields=[
                    "Th" => cache_fefunc[1], 
                    "uh" => cache_fefunc[3],
                    "χ" => cache_coeff[1]
                ]
            )
        end 
        with_logger(tb_lg) do 
            image_χ = TBImage(cache_χ, WH)
            @info "energy" E1= J[1] E2= J[2] E3= J[3] E= E
            @info "domain" χ= image_χ log_step_increment=0
            @info "parameters" τ= τ  ϵ= curϵ log_step_increment=0
            @info "count" in_iter= n_in_iter in_time= time_in out_time= time_out volₖ= volₖ M= M log_step_increment=0
        end
        if curϵ < ϵ
            update_tau!(motion, ϵ_ratio)
        end

        i += 1
    end

    close(tb_lg);
    writevtk(Ω, joinpath(vtk_prefix, "result"); 
            cellfields=[
                "Th" => cache_fefunc[1], 
                "Thˢ" => cache_fefunc[2],
                "uh" => cache_fefunc[3],
                "uhˢ" => cache_fefunc[4],
                "χ" => cache_coeff[1]
            ]
        )
    
    return χ₀, cache_χ
end

"""
run with vectors of pairs of configure.
"""
function run_with_configs(vec_configs::Vector, comments)
    idx = findfirst(((k, _),) -> k == "max_it", vec_configs)
    max_it = last(vec_configs[idx])

    multi_config_pairs = filter(((_, v),) -> isa(v, Vector), vec_configs) 
    single_config_pairs = filter(((_, v),) -> !isa(v, Vector), vec_configs)
        
    base_config = Dict(single_config_pairs)
    broadcasted_config_pairs = map(((k, v),) -> broadcast(Pair, k, v), multi_config_pairs)
    appended_config_arr = Iterators.product(broadcasted_config_pairs...) |> collect

    # data path 
    path = joinpath("data", string(now()))
    mkpath(path)

    open(joinpath(path, "vec_configs.jl"), "w") do io
        println(io, vec_configs)
    end
    
    # info
    dict_info = Dict(
        "HOSTNAME" => ENV["HOSTNAME"],
        "LOGNAME" => ENV["LOGNAME"],
        "PWD" => ENV["PWD"],
        "PID" => getpid(),
        "NPROCS" => nprocs(),
        "COMMENTS" => comments
    )
    open(joinpath(path, "config.toml"), "w") do io
        out = Dict(
            "info" => dict_info,
            "vec_configs" => Dict(vec_configs),
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
    mkpath(tb_path_prefix)

    sh_file_name = joinpath(path, "tensorboard.sh")
    touch(sh_file_name)
    chmod(sh_file_name, 0o777)
    open(sh_file_name, "w") do io
        println(io, "#!/bin/bash")
        println(io, "tensorboard --logdir=$(tb_path_prefix) --port=\$1 --samples_per_plugin=images=$(max_it+1)")
    end

    jld2_prefix = joinpath(path, "jld2")
    mkpath(jld2_prefix)

    @sync @distributed for i = eachindex(appended_config_arr)
        multi_config = Dict(appended_config_arr[i])
        all_config = merge(base_config, multi_config)

        tb_file = joinpath(tb_path_prefix, "run_$i")

        tb_lg = TBLogger(tb_file, tb_overwrite, min_level=Logging.Info)
        if !isempty(multi_config)
            write_hparams!(tb_lg, multi_config, ["energy/E"])
        end
        with_logger(tb_lg) do 
            @info "config" base=TBText(DataFrame(base_config)) log_step_increment=0
        end

        vtk_prefix = joinpath(path, "vtk", "run_$i")
        mkpath(vtk_prefix)
       
        χ₀, χ = singlerun(all_config, vtk_prefix, tb_lg)

        jldopen(joinpath(jld2_prefix, "run_$i.jld2"), "w") do _file
            _file["χ₀"] = χ₀
            _file["χ"] = χ
            _file["config"] = all_config
        end
    end
    return nothing
end