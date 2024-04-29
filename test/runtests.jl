using Distributed

# run at 138
addprocs([
            ("c95", 8),
            ("c0130", 8),
            ("c0123", 4),
            ("c0124", 4),
        ], 
        tunnel= true,
        enable_threaded_blas= true,
        topology=:master_worker,
        exeflags="--project"
    )

@everywhere using HeatFlowTopOpt

vec_configs =[
    # pde parameter
    "β₁" => [0., 0.1, 10.],
    "β₂" => 1.,
    "β₃" => 1,
    "δt" => 8e-3,
    "α⁻" => [417.5, 41750.],
    "Re" => 5988.,
    "ud⋅n" => 0.,
    "Td" => 0.0,
    "g⋅n" => [10., 33.5, 50.],
    "Ts" => 1.,

    # motion paramter
    "up" => 0.95,
    "down" => 0.05,
    "τ₀" => [1e-3, 1e-4],
    "motion_tag" => "conv",

    # top opt parameter
    "correct_ratio" => 0.5,
    "ϵ_ratio" => 0.5,
    "ϵ" => 10., 
    "save_iter" => 10,
    "vol" => 0.4,
    "max_it" => 1000,
    "InitType" => ["Net", "Line"],

    # model parameter
    "N" => 254, # cell
    "dim" => 2,
    "L" => 1.
];
comments = "测试β₁, α⁻, g⋅n, 以及初始区域（网状/条状）的结果。"

run_with_configs(vec_configs, comments)
