using Distributed

# # run at 138
addprocs(
        [
            ("c95", 2),
            ("c0130", 2),
            # ("yhxiang@197", 8),
            # ("c0123", 8),
            # ("c0124", 8),
        ], 
        # 2,
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

# @everywhere using MKL
@everywhere using HeatFlowTopOpt

vec_configs = [
    # pde parameter
    "β₁" => 0.,
    "β₂" => 0.,
    "β₃" => 1,
    "δt" => 8e-3,
    "δu" => 5e-2,
    "α⁻" => 417.5,
    "kf" => 0.1624,
    "ks" => 40.07,
    "Re" => 5988.,
    "γ" => 1027.6,
    "Ts" => 1.,
    "ud⋅n" => 0.,
    "Td" => 0.0,
    "g⋅n" => 30.,
    "Ts" => 1.,

    # motion paramter
    "up" => 0.95,
    "down" => 0.05,
    "τ₀" => 5e-4,
    "motion_type" => "conv",
    "pdsz" => 1,
    "lr" => [0.1, 1e-3],

    # top opt parameter
    "correct_rate" => 0.5,
    "ϵ_ratio" => 0.5,
    "ϵ" => 20., 
    "save_iter" => 500,
    "save_start" => 0,
    "vol" => 0.3,
    "max_it" => 500,
    "InitType" => ["Rand", "Lines"],
    "InitFile" => "",
    "InitKey" => "",
    "scheme" => SCHEME_GRADIENT,
    "rand_rate" => 0.5,
    "rand_kernel_dim" => 4,

    # model parameter
    "Nc" => 600,  # 240 for Line initialization, 240 ÷ 2 ÷ 20
    "dim" => 2,
    "L" => 1.
];
comments = "debug"

run_with_configs(vec_configs, comments)
