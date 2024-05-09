using Distributed

# # run at 138
addprocs([
            ("c95", 16),
            ("c0130", 16),
            ("yhxiang@197", 16),
            ("c0123", 8),
            ("c0124", 8),
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
    "β₂" => [0.1, 10.],
    "β₃" => 1,
    "δt" => 8e-3,
    "α⁻" => [417.5, 4175.],
    "α₋" => 0.,
    "kf" => 0.1624,
    "ks" => 40.47,
    "Re" => 5988.,
    "γ" => 1027.6,
    "Ts" => 1.,
    "ud⋅n" => 0.,
    "Td" => 0.0,
    "g⋅n" => [10., 50.],
    "Ts" => 1.,

    # motion paramter
    "up" => 0.95,
    "down" => 0.05,
    "τ₀" => [1e-3, 1e-4],
    "motion_tag" => "conv",

    # top opt parameter
    "correct_ratio" => 0.5,
    "ϵ_ratio" => 0.5,
    "ϵ" => 20., 
    "save_iter" => 10,
    "vol" => [0.2, 0.4],
    "max_it" => 1000,
    "InitType" => ["Rand", "Line"],
    "is_correct" => [true, false],
    "is_restart" => false,       # if not correction, then restart is off.
    "is_vol_constraint" => true, # if false, then set val to a scalar.
    "is_bdupdate" => false,

    # model parameter
    "N" => 240,  # 240 for Line initialization, 240 ÷ 2 ÷ 20
    "dim" => 2,
    "L" => 1.
];
comments = "利用ICTM的更新方式，进行调参。"

run_with_configs(vec_configs, comments)
