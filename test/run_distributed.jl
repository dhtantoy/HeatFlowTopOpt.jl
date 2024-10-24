ENV["JULIA_CPU_TARGET"] = "generic;icelake-server,clone_all;skylake-avx512,clone_all;skylake,clone_all"
using HeatFlowTopOpt
using Distributed
# # run at 138
addprocs(
        [
            ("c95", 1),
            ("c0130",1),
            # ("yhxiang@197", 1),
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
    "η" => 0.,
    "α⁻" => [1e4, 1e6],
    "E" => 3e9,
    "θ" => 0.3,
    "Emin" => 1e-9,
    "ν" => 1e-3,
    "Vdₓ" => 1.,

    # motion paramter
    "up" => 0.95,
    "down" => 0.05,
    "τ₀" => 1e-4,
    "motion_type" => "conv",
    "pdsz" => 3,

    # top opt parameter
    "correct_rate" => 0.5,
    "ϵ_ratio" => 0.5,
    "ϵ" => 10., 
    "save_iter" => 50,
    "save_start" => 0,
    "vol" => 0.1,
    "max_it" => 50,
    "InitType" => "Rand",
    "InitFile" => "",
    "InitKey" => "",
    "scheme" => [SCHEME_NULL, SCHEME_CORRECT],
    "rand_rate" => 0.5,
    "rand_kernel_dim" => 4,

    # model parameter
    "Nc" => 200 * 0.6,  #
    "ModelFile" => "ela_flow"
];
comments = "ela-flow model"

run_with_configs(vec_configs, comments)
