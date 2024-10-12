ENV["JULIA_CPU_TARGET"] = "generic;icelake-server,clone_all;skylake-avx512,clone_all;skylake,clone_all"
using HeatFlowTopOpt
using Distributed
# # run at 138
addprocs(
        [
            ("c95", 1),
            ("c0130",1),
            ("yhxiang@197", 1),
            ("c0123", 1),
            ("c0124", 1),
        ], 
        # 2,
        tunnel= true,
        enable_threaded_blas= true,
        topology=:master_worker,
        exeflags="--project"
    )


@everywhere using HeatFlowTopOpt

vec_configs = [
    # pde parameter
    "β₁" => 0.,
    "β₂" => 0.,
    "β₃" => 1,
    "δt" => 0.,
    "α⁻" => [100, 200],
    "kf" => 0.,
    "ks" => 10,
    "Re" => [300, 400, 500],
    "Pr" => 6.7,
    "Ts" => 1.,
    "Vdₓ" => 0.,
    "Td" => 0.0,
    "Pd" => 1.,
    "Ts" => 1.,

    # motion paramter
    "up" => 0.95,
    "down" => 0.05,
    "τ₀" => 1e-4,
    "motion_type" => "conv",
    "pdsz" => 3,

    # top opt parameter
    "correct_rate" => 0.8,
    "ϵ_ratio" => 0.5,
    "ϵ" => 10., 
    "save_iter" => 100,
    "save_start" => 0,
    "vol" => [0.6, 0.7],
    "max_it" => 100,
    "InitType" => ["Line", "Lines", "Rand"],
    "InitFile" => "",
    "InitKey" => "",
    "scheme" => SCHEME_NULL,
    "rand_rate" => 0.5,
    "rand_kernel_dim" => 4,

    # model parameter
    "Nc" => 300,  # 240 for Line initialization, 240 ÷ 2 ÷ 20
    "ModelFile" => "lcr_model"
];
comments = "test newer config"

run_with_configs(vec_configs, comments)
