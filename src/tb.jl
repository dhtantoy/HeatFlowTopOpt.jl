@inline sigmod(x) = 1/(1 + exp(2(x - 10)))

function singlerun(config, it, tb_lg, file_jld2; debug= true, vtk_path= "debug")
    β₁ = config["β₁"]
    β₂ = config["β₂"]
    β₃ = config["β₃"]
    α⁻ = config["α⁻"]
    N::Int = config["N"]
    up = config["up"]
    down = config["down"]
    ϵ = config["ϵ"]
    Re = config["Re"]
    ϵ_ratio = config["ϵ_ratio"] 
    correct_ratio = config["correct_ratio"];
    save_iter::Int = config["save_iter"]
    τ₀ = config["τ₀"]
    ud = VectorValue(config["ud⋅n"], 0.)
    g = VectorValue(config["g⋅n"], 0.)
    Td = config["Td"]
    vol = config["vol"]

    conv = Conv(Float64, 2, N + 1, τ₀; time= 120);
    cache_Gτχ = similar(conv.out)

    # (m, Ω, Γ, lab, dx, dσ, χ, aux_space)
    m, Ω, _, lab, dx, dσ, cache_χ, aux_space = initmodel(N);
    volₖ = sum(cache_χ) / length(cache_χ);

    # spaces = (T_test, T_trial, X, Y)
    # cache_fem = (T_assem, V_assem, cache_T_b, cache_T_A, cache_V_b, cache_V_A)
    # cache_fefunc = (cache_Th, cache_Thˢ, cache_uh, cache_uhˢ)
    spaces, cache_fem, cache_fefunc = initspace(m, lab, dx; ud= ud, Td= Td);
    
    # allocate cache for computing nodal value and Φ
    # (cache_Φ, cache_rev_Φ, cache_node_val)
    cache_Φ = Matrix{Float64}(undef, N + 1, N + 1);
    cache = (cache_Φ, similar(cache_Φ), similar(cache_Φ));
    
    # allocate cache for χ_old
    χ_old = similar(cache_χ);

    # # save data
    file_jld2["χ_0"] = cache_χ
    
    # fem params 
    params = (g, β₁, β₂, β₃, N, Re, α⁻)
    
    δt = config["δt"]

    @info @green "computing initial energy ..."
    # return J
    cache_coeff = _coeff_cache(cache_χ, conv, aux_space, α⁻)
    J = fem_solve!(cache_fem, cache_fefunc, cache_coeff, params, spaces, dx, dσ, conv, δt)
    E = +(J...);

    n = (N+1)^2;
    τ = τ₀

    # allocate cache for iteration and correction
    cache_val_exist = SzVector(Float64, n, 0);
    cache_idx_exist = SzVector(Float64, n, 0);
    idx_exist_sort_pre = SzVector(Int, n, 0);
    idx_pick_post = SzVector(Int, n, 0);
    idx_increase = SzVector(Int, n, 0);
    idx_decrease = SzVector(Int, n, 0);

    @info "----------------- start iteration ------------------"


    function F()
        i = 1
        while i <= it
            @info @green "iteration $(i): "

            time_out = time()
            Nₖ = sum(cache_Gτχ)
            copy!(χ_old, cache_χ);
            copy!(cache_Gτχ, conv.out);

            fem_adjoint_solve!(cache_fem, cache_fefunc, cache_coeff, params, spaces, dx, δt)
            Phi!(cache, params, cache_fefunc, Ω, conv)
            n_geq_1, n_leq_0 = post_phi!(cache_Φ, cache_Gτχ, down, up)

            # vol_update_r = (vol - volₖ) / (vol + volₖ)
            # n_in_term = Nₖ - n_geq_1
            # n_out_term = length(cache_χ) - n_leq_0 - n_in_term
            # M1 = min(n_in_term, n_out_term) * vol_update_r + Nₖ |> floor
            M1 = length(cache_χ) * vol
            M2 = count(<=(0), cache_Φ)
            # λ = ((τ - 1e-6)/(1e-3 - 1e-6))^(1/4)
            λ = sigmod(log2(τ₀ / τ))

            M::Int = M2 * λ + M1 * (1 - λ) |> floor

            get_ordered_idx!(idx_exist_sort_pre, cache_val_exist, cache_idx_exist, cache_χ, cache_Φ);

            iterateχ!(cache_χ, cache_Φ, idx_pick_post, M);

            # return Th, Thˢ, uh, uhˢ, nothing/J
            cache_coeff = _coeff_cache(cache_χ, conv, aux_space, α⁻)
            J = fem_solve!(cache_fem, cache_fefunc, cache_coeff, params, spaces, dx, dσ, conv, δt)
            Ei = +(J...)

            time_out = time() - time_out
            time_in = time()
            
            # correction
            n_in_iter = 0
            err_flag_in_iter = false
            while Ei >= E
                n_in_iter += 1

                n_in_iter > 50 && error("n_in_iter too big!")

                computeAB!(idx_decrease, idx_increase, idx_exist_sort_pre, idx_pick_post, cache_Φ)

                if length(idx_decrease) == 0 && length(idx_increase) == 0 
                    @info @red "correction failed! now restart with a smaller τ." 
                    err_flag_in_iter = true
                    break
                end
                correct!(cache_χ, correct_ratio, idx_decrease, idx_increase, cache_Φ)

                get_ordered_idx!(idx_pick_post, cache_val_exist, cache_idx_exist, cache_χ, cache_Φ)
                
                cache_coeff = _coeff_cache(cache_χ, conv, aux_space, α⁻)
                J = fem_solve!(cache_fem, cache_fefunc, cache_coeff, params, spaces, dx, dσ, conv, 8e-3)
                Ei = +(J...)
            end

            τ = conv.τ[]

            if τ < 1e-8 
                @info @red "τ < 1e-8 and break iteration."
                break
            end
            
            # restar with a smaller τ
            if err_flag_in_iter
                update_tau!(conv, ϵ_ratio);
                # restore cache data
                copy!(cache_χ, χ_old);
                copy!(conv.out, cache_Gτχ)
                continue
            end

            E = Ei
            time_in = time() - time_in

            volₖ = sum(cache_χ) / length(cache_χ)
            curϵ = norm(cache_χ - χ_old, 2)
            @info "J = $(J), E = $E, τ = $(τ), cur_ϵ= $(curϵ), β₂ = $β₂, in_iter= $n_in_iter"
            if mod(i, save_iter) == 0 
                file_jld2["χ_$(i)"] = cache_χ
                writevtk(Ω, joinpath(vtk_path, string(i)); cellfields=["Th" => cache_fefunc[1], "uh" => cache_fefunc[3]])
            end 
            if !debug 
                with_logger(tb_lg) do 
                    image_χ = TBImage(cache_χ, WH)
                    @info "energy" E1= J[1] E2= J[2] E3= J[3] E= E log_step_increment=0
                    @info "domain" χ= image_χ log_step_increment=0
                    @info "parameters" τ= τ  ϵ= curϵ log_step_increment=0
                    @info "count" in_iter= n_in_iter in_time= time_in out_time= time_out volₖ= volₖ M= M λ= λ
                end
            else
                Plots.heatmap(cache_χ) |> display
            end
            if curϵ < ϵ
                update_tau!(conv, ϵ_ratio)
            end

            i += 1
        end
    end

    
    F()
    close(tb_lg);
    writevtk(Ω, joinpath(vtk_path, "result"); cellfields=["Th" => cache_fefunc[1], "Thˢ" => cache_fefunc[2], "uh" => cache_fefunc[3], "uhˢ" => cache_fefunc[4]])
    nothing
end

function run_with_configs(configs::Vector; it= 1000, debug= false)
    multi_config_pairs = filter(((_, v),) -> isa(v, Vector),  configs) 
    single_config_pairs = filter(((_, v),) -> !isa(v, Vector),  configs)
        
    base_config = Dict(single_config_pairs)
    broadcasted_config_pairs = map(((k, v),) -> broadcast(Pair, k, v), multi_config_pairs)
    appended_config_arr = Iterators.product(broadcasted_config_pairs...)

    # data path 
    path = joinpath("data", string(now()))
    mkpath(path)

    # info
    HOSTNAME = ENV["HOSTNAME"]
    LOGNAME = ENV["LOGNAME"]
    PWD = ENV["PWD"]
    PID = getpid()
    TB_PORT = get_avaible_port()
    open(joinpath(path, "info.txt"), "w") do io
        println(io, "HOSTNAME: $HOSTNAME")
        println(io, "LOGNAME: $LOGNAME")
        println(io, "PWD: $PWD")
        println(io, "PID: $PID")
        println(io, "MAX ITER: $it")
        println(io, "TB_PORT: $TB_PORT")
        println(io, "DEBUG: $debug")
    end

    # tensorboard script
    tb_path = joinpath(PWD, path, "tb")
    mkpath(tb_path)
    
    # @spawn run(`tensorboard --logdir=$(tb_path) --port=$TB_PORT --samples_per_plugin=images=$it`; wait= false)
    tb_proc = run(`tensorboard --logdir=$(tb_path) --port=$TB_PORT --samples_per_plugin=images=$it`; wait= false)

    sh_file_name = joinpath(path, "tensorboard.sh")
    touch(sh_file_name)
    chmod(sh_file_name, 0o777)
    open(sh_file_name, "w") do io
        println(io, "#!/bin/bash")
        println(io, "tensorboard --logdir=$(tb_path) --port=\$1 --samples_per_plugin=images=$it --load_fast=false")
    end

    @info @yellow "data will be stored in $(path)"
    for (i, c) = enumerate(appended_config_arr)
        multi_config = Dict(c)
        all_config = merge(base_config, multi_config)

        jld2_dir = joinpath(path, "jld2")
        mkpath(jld2_dir)
        file_jld2 = jldopen(joinpath(jld2_dir, "$i.jld2"), "w")
        file_jld2["config"] = all_config

        file_name_tb = joinpath(tb_path, string(i))
        tb_lg = TBLogger(file_name_tb, tb_overwrite, min_level=Logging.Info)
        if !isempty(multi_config)
            write_hparams!(tb_lg, multi_config, ["energy/E"])
        end

        with_logger(tb_lg) do 
            @info "config" base=TBText(DataFrame(base_config)) log_step_increment=0
            @info "config" multi=TBText(DataFrame(multi_config)) 
        end

        vtk_path = joinpath(path, "vtk")
        mkpath(vtk_path)
        singlerun(all_config, it, tb_lg, file_jld2; debug= debug, vtk_path= vtk_path)
    end

    kill(tb_proc)
    return nothing
end