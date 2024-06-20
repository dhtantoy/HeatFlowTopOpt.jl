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
        correct_ratio = config["correct_ratio"]
        save_iter::Int = config["save_iter"]
        save_start::Int = config["save_start"]
        vol = config["vol"]
        is_correct = config["is_correct"]
        is_restart = config["is_restart"]
        is_vol_constraint = config["is_vol_constraint"]
        is_bdupdate = config["is_bdupdate"]

        # parameters corresponding to motion
        up = config["up"]
        down = config["down"]
        τ₀ = config["τ₀"]
        InitType = config["InitType"]
        motion_tag = config["motion_tag"]
        
        # parameters corresponding to PDE
        β₁ = config["β₁"]
        β₂ = config["β₂"]
        β₃ = config["β₃"]
        α⁻ = config["α⁻"]
        α₋ = config["α₋"]
        kf = config["kf"]
        ks = config["ks"]
        γ = config["γ"]
        Ts = config["Ts"]
        Re = config["Re"]
        δt = config["δt"]
        ud = VectorValue(config["ud⋅n"], 0.)
        g = VectorValue(config["g⋅n"], 0.)
        Td = config["Td"]
        dim = config["dim"]
        L = config["L"]
        
        if motion_tag == "conv"
            motion = Conv(Float64, 2, N + 1, τ₀; time= 120);
        else
            error("motion not defined!")
        end

        τ = τ₀
    end

    # ----------------------------------- Host output -----------------------------------
    with_logger(tb_lg) do 
        dict_info = Dict(
            "LOGNAME" => ENV["LOGNAME"],
            "PID" => getpid(),
            "UID" => parse(Int, readchomp(`id -u`)),
            "GID" => parse(Int, readchomp(`id -g`)),
        )   
        try 
            push!(dict_info, "HOSTNAME" => ENV["HOSTNAME"])
        catch
        end
        @info "host" base=TBText(DataFrame(dict_info)) log_step_increment=0
    end

    # ----------------------------------- model setting -----------------------------------
    # init gmsh model
    ## "wall"-1, "inlet"-2, "outlet"-3, "block_left"-4, "block_center"-5, "block_right"-6, "body"-7
    # file = joinpath("models", "N_$(N)_L_$L.msh")
    # if !isfile(file)
    #     @assert iszero(N % 3) "N should be divided by 3."
    #     createGrid(Val(dim), N ÷ 3, L, file)
    # end
    # model = DiscreteModelFromFile(file)

    ## CartesianDiscreteModel
    ## "inlet" - 7, "outlet" - 8, "wall" - [5, 6], "motion_domain" - ["interior"], "fixed_domain" - []
    diri_tags = [[7], [5, 6]]
    model = CartesianDiscreteModel(repeat([0, L], dim), repeat([N], dim)) |> simplexify

    # bd_tag = ["wall", "inlet", "outlet"]
    (motion_space, fixed_space), (Ω, Γs), (dx, dΓs), perm = initmodel(model, ["interior"], [], [[5, 6], 7, 8]);
    fixed_fe_χ = FEFunction(fixed_space, Fill(1., num_free_dofs(fixed_space)));
    # ---------------------------------------------------------------------------------------------------------
    
    
    # init χs and prepare cache for array.
    motion_cache_arr_χ, motion_cache_arr_χ₂, motion_cache_arr_Gτχ, motion_cache_arr_Gτχ₂ = initcachechis(InitType, motion_space; vol= vol);
    volₖ = sum(motion_cache_arr_χ) / length(motion_cache_arr_χ);
    M = round(Int, length(motion_cache_arr_χ) * vol);
    χ₀ = copy(motion_cache_arr_χ);
    motion_arr_χ_old = similar(motion_cache_arr_χ);

    # init fe funcs 
    motion_cache_fe_funcs = initfefuncs(motion_space);

    # init spaces
    test_spaces, trial_spaces, assemblers, cache_As, cache_bs, cache_fe_funcs, cache_ad_fe_funcs = initspaces(model, dx, Td, ud, diri_tags)
    
    # allocate cache for computing nodal value and Φ
    # (cache_Φ, cache_rev_Φ, cache_node_val)
    motion_cache_Φ = Matrix{Float64}(undef, N + 1, N + 1);
    motion_cache_Φs = (motion_cache_Φ, similar(motion_cache_Φ), similar(motion_cache_Φ));
    
    @info "run_$(run_i): computing initial energy ..."
    motion_cache_arrs = (motion_cache_arr_χ, motion_cache_arr_χ₂, motion_cache_arr_Gτχ, motion_cache_arr_Gτχ₂);
    params = (α₋, α⁻, kf, ks)
    coeffs = getcoeff!(motion_cache_arrs, motion_cache_fe_funcs, fixed_fe_χ, motion, params, perm);

    cur_δt = δt*(1 - coeffs[2])
    params = (g, β₁, β₂, β₃, N, Re, cur_δt, γ, Ts, motion.τ[])
    J = pde_solve!(cache_fe_funcs, test_spaces, trial_spaces, cache_As, cache_bs, assemblers,
                params, coeffs, dx, dΓs)
    E = +(J...);

    with_logger(tb_lg) do 
        image_χ = TBImage(motion_cache_arr_χ, WH)
        @info "energy" E1= J[1] E2= J[2] E3= J[3] E= E
        @info "domain" χ= image_χ log_step_increment=0
    end

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
        @info "run_$(run_i) iteration $(i): "

        time_out = time()
        copy!(motion_arr_χ_old, motion_cache_arr_χ);

        cur_δt = δt*(1 - coeffs[2])
        params = (N, Re, cur_δt, γ, β₃)
        adjoint_pde_solve!(cache_ad_fe_funcs, cache_fe_funcs, test_spaces, trial_spaces, cache_As, cache_bs, assemblers,
                        params, coeffs, dx)

        # all pde solved.
        if i >= save_start && mod(i, save_iter) == 0
            Th, uh = cache_fe_funcs
            fe_χ = coeffs[1]
            if debug
                Thˢ, uhˢ = cache_ad_fe_funcs
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
                    "δt" => cur_δt,
                    "uh⋅uhˢ" => uh⋅uhˢ,
                    "∇(Th)⋅∇(Thˢ)" => ∇(Th)⋅∇(Thˢ),
                    "(Th - Ts) * Thˢ" => (Ts - Th) * Thˢ,
                    "-Re*Thˢ*∇Th" => -Re * Thˢ * ∇(Th),

                    "β₁/2 * (α⁻ - α₋) * uh⋅uh" => β₁/2 * (α⁻ - α₋) * uh⋅uh,
                    "β₃ * γ * (kf - ks) * (Ts - Th)" => β₃ * γ * (kf - ks) * (Ts - Th),
                    "(α⁻ - α₋) * (uh⋅uhˢ)" => (α⁻ - α₋) * (uh⋅uhˢ),
                    "(ks - kf) * ∇(Th)⋅∇(Thˢ)" => (ks - kf) * ∇(Th)⋅∇(Thˢ),
                    "γ * (ks - kf) * (Th - Ts) * Thˢ" => γ * (ks - kf) * (Th - Ts) * Thˢ,
                    "γ * (ks - kf) * (Th - Ts) * (β₃ + Thˢ)" => γ * (ks - kf) * (Th - Ts) * (β₃ + Thˢ),
                    "Φ" => β₁/2 * (α⁻ - α₋) * uh⋅uh + 
                            β₃ * γ * (kf - ks) * (Ts - Th) + 
                            (α⁻ - α₋) * (uh⋅uhˢ) + 
                            (ks - kf) * ∇(Th)⋅∇(Thˢ) + 
                            γ * (ks - kf) * (Th - Ts) * Thˢ,
                    ]
            else
                c_fs = ["Th" => Th, "uh" => uh, "χ" => fe_χ]
            end
            vtk_file_pvd[Float64(i)] =  createvtk(Ω, vtk_file_prefix * string(i); cellfields= c_fs)
        end

        params = (β₁, β₂, β₃, α⁻, α₋, Ts, kf, ks, γ)
        Phi!(motion_cache_Φs, params, cache_fe_funcs, cache_ad_fe_funcs, motion_space, motion, motion_cache_arr_Gτχ, motion_cache_arr_Gτχ₂)
        
        # update on the boundary
        if is_bdupdate
            post_phi!(motion_cache_Φ, motion_cache_arr_Gτχ, down, up)
        end

        # volumn constraint
        if !is_vol_constraint
            M = count(<=(0), motion_cache_Φ)
        end

        get_ordered_idx!(idx_exist_sort_pre, cache_val_exist, cache_idx_exist, motion_cache_arr_χ, motion_cache_Φ);

        iterateχ!(motion_cache_arr_χ, motion_cache_Φ, idx_pick_post, M);

        params = (α₋, α⁻, kf, ks)
        coeffs = getcoeff!(motion_cache_arrs, motion_cache_fe_funcs, fixed_fe_χ, motion, params, perm)

        cur_δt = δt*(1 - coeffs[2])
        params = (g, β₁, β₂, β₃, N, Re, cur_δt, γ, Ts, motion.τ[])
        J = pde_solve!(cache_fe_funcs, test_spaces, trial_spaces, cache_As, cache_bs, assemblers,
                    params, coeffs, dx, dΓs)
        Ei = +(J...);

        time_out = time() - time_out
        
        time_in = time()
        n_in_iter = 0
        err_flag_in_iter = false
        
        # correction
        if is_correct
            while Ei >= E
                n_in_iter += 1

                n_in_iter > 20 && error("n_in_iter too big!")

                computeAB!(idx_decrease, idx_increase, idx_exist_sort_pre, idx_pick_post, motion_cache_Φ)

                if length(idx_decrease) == 0 && length(idx_increase) == 0 
                    @info "run_$(run_i): correction failed! now restart with a smaller τ." 
                    err_flag_in_iter = true
                    break
                end
                correct!(motion_cache_arr_χ, correct_ratio, idx_decrease, idx_increase, motion_cache_Φ)

                get_ordered_idx!(idx_pick_post, cache_val_exist, cache_idx_exist, motion_cache_arr_χ, motion_cache_Φ)
                
                params = (α₋, α⁻, kf, ks)
                coeffs = getcoeff!(motion_cache_arrs, motion_cache_fe_funcs, fixed_fe_χ, motion, params, perm)

                cur_δt = δt*(1 - coeffs[2])
                params = (g, β₁, β₂, β₃, N, Re, cur_δt, γ, Ts, motion.τ[])
                J = pde_solve!(cache_fe_funcs, test_spaces, trial_spaces, cache_As, cache_bs, assemblers,
                            params, coeffs, dx, dΓs)
                Ei = +(J...)
            end
        end

        τ = motion.τ[]
        if τ < 1e-8 
            @info "run_$(run_i): τ < 1e-8 and break iteration."
            break
        end
        
        # restar with a smaller τ
        if err_flag_in_iter
            if is_restart
                update_tau!(motion, ϵ_ratio);
                # restore cache data
                copy!(motion_cache_arr_χ, motion_arr_χ_old);
                continue
            else
                @info "run_$(run_i): restarting is off, now break iteration."
                break
            end
        end

        E = Ei
        time_in = time() - time_in

        volₖ = sum(motion_cache_arr_χ) / length(motion_cache_arr_χ)
        curϵ = norm(motion_cache_arr_χ - motion_arr_χ_old, 2)
        @info "run_$(run_i): J = $(J), E = $E, τ = $(τ), cur_ϵ= $(curϵ), β₂ = $β₂, in_iter= $n_in_iter"
        with_logger(tb_lg) do 
            image_χ = TBImage(motion_cache_arr_χ, WH)
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

    vtk_file_pvd[Float64(max_it + 1)] = createvtk(Ω, vtk_file_prefix * string(max_it + 1); 
            cellfields=[
                "Th" => cache_fe_funcs[1], 
                "Thˢ" => cache_ad_fe_funcs[1],
                "uh" => cache_fe_funcs[2],
                "uhˢ" => cache_ad_fe_funcs[2],
                "χ" => coeffs[1]
            ]
        )

    return χ₀, motion_cache_arr_χ
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
    make_path(tb_path_prefix, 0o753)

    sh_file_name = joinpath(path, "tensorboard.sh")
    touch(sh_file_name)
    chmod(sh_file_name, 0o755)
    open(sh_file_name, "w") do io
        println(io, "#!/bin/bash")
        println(io, "tensorboard --logdir=$(tb_path_prefix) --port=\$1 --samples_per_plugin=images=$(max_it+1)")
    end

    jld2_prefix = joinpath(path, "jld2")
    make_path(jld2_prefix, 0o753)

    vtk_prefix = joinpath(path, "vtk")
    make_path(vtk_prefix, 0o753)

    pmap(eachindex(appended_config_arr)) do i
        multi_config = Dict(appended_config_arr[i])
        config = merge(base_config, multi_config)

        tb_file_prefix = joinpath(tb_path_prefix, "run_$i")
        tb_lg = TBLogger(tb_file_prefix, tb_overwrite, min_level=Logging.Info)

        if !isempty(multi_config)
            write_hparams!(tb_lg, multi_config, ["energy/E"])
        end

        vtk_file_prefix = joinpath(vtk_prefix, "run_$(i)_")
        vtk_file_pvd = createpvd(vtk_file_prefix)
       
        χ₀, χ = singlerun(config, vtk_file_prefix, vtk_file_pvd, tb_lg, i)

        jld2_file = joinpath(jld2_prefix, "run_$i.jld2")
        jldopen(jld2_file, "w") do _file
            _file["χ₀"] = χ₀
            _file["χ"] = χ
            _file["config"] = config
        end

        close(tb_lg);
        savepvd(vtk_file_pvd);

        nothing
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