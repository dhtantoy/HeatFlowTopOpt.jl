using Distributed

# # run at 138
addprocs([
            ("c95", 3),
            ("c0130", 3),
            ("c0123", 6),
            ("c0124", 6),
        ], 
        tunnel= true,
        enable_threaded_blas= true,
        topology=:master_worker,
        exeflags="--project"
    )

@everywhere using HeatFlowTopOpt

vec_configs =[
    # pde parameter
    "β₁" => 0.,
    "β₂" => 5e-2,
    "β₃" => 1,
    "δt" => 8e-3,
    "α⁻" => 417.5,
    "α₋" => 0.,
    "kf" => 0.1624,
    "ks" => 40.47,
    "Re" => 5988.,
    "γ" => 1027.6,
    "Ts" => 1.,
    "ud⋅n" => 0.,
    "Td" => 0.0,
    "g⋅n" => 33.5,
    "Ts" => 1.,

    # motion paramter
    "up" => 0.95,
    "down" => 0.05,
    "τ₀" => 3e-4,
    "motion_tag" => "conv",

    # top opt parameter
    "correct_ratio" => 0.5,
    "ϵ_ratio" => 0.5,
    "ϵ" => 20., 
    "save_iter" => 10,
    "vol" => [0.4, 0.8],
    "max_it" => 1000,
    "InitType" => ["All", "Rand"],
    "is_correct" => [true, false],
    "is_restart" => [true, false],
    "is_vol_constraint" => true, # if false, then set val to a scalar.
    "is_bdupdate" => [true, false],

    # model parameter
    "N" => 240, # cell
    "dim" => 2,
    "L" => 1.
];
comments = "修正了卷积的问题后重新测试，本测试有体积约束."

run_with_configs(vec_configs, comments)
