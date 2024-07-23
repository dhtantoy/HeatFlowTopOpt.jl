using Distributed

# # run at 138
addprocs([
            ("c95", 8),
            # ("c0130", 2),
            # ("yhxiang@197", 2),
            # ("c0123", 4),
            # ("c0124", 4),
        ], 
        tunnel= true,
        enable_threaded_blas= true,
        topology=:master_worker,
        exeflags="--project"
    )

# # interrupt the process when exit
atexit() do 
    for p in workers()
        rmprocs(p)
    end
end

@everywhere using MKL
@everywhere using HeatFlowTopOpt


vec_configs = [
    # pde parameter
    "β₁" => 0.,
    "β₂" => 1.,
    "β₃" => 1,
    "δt" => 8e-3,
    "δu" => 5e-2,
    "α⁻" => 417.5,
    "α₋" => 0.,
    "kf" => 0.1624,
    "ks" => 40.47,
    "Re" => 5988.,
    "γ" => 1027.6,
    "Ts" => 1.,
    "ud⋅n" => 0.,
    "Td" => 0.0,
    "g⋅n" => [0.1, 33.5],
    "Ts" => 1.,

    # motion paramter
    "up" => 0.95,
    "down" => 0.05,
    "τ₀" => 7e-4,
    "motion_tag" => "conv",

    # top opt parameter
    "correct_ratio" => 0.5,
    "ϵ_ratio" => 0.5,
    "ϵ" => 40., 
    "save_iter" => 30,
    "save_start" => 0,
    "vol" => 0.3,
    "max_it" => 1000,
    "InitType" => ["Rand", "Line"],
    "is_correct" => true,
    "is_vol_constraint" => true, # if false, then set val to a scalar.
    "is_bdupdate" => [false, true],

    # model parameter
    "N" => 240,  # 240 for Line initialization, 240 ÷ 2 ÷ 20
    "dim" => 2,
    "L" => 1.
];
comments = "testing..."

run_with_configs(vec_configs, comments)
