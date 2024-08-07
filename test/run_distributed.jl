using Distributed

# # run at 138
addprocs(
        [
            ("c95", 4),
            ("c0130", 4),
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
    "β" => [20, 50],
    "q1" => 1.,
    "q2" => 100.,
    "k1" => [10., 50., 100.],
    "k2" => 1.,

    # motion paramter
    "up" => 0.95,
    "down" => 0.05,
    "τ₀" => 1e-4,
    "motion_type" => "conv",

    # top opt parameter
    "correct_rate" => 0.5,
    "ϵ_ratio" => 0.5,
    "ϵ" => 10., 
    "save_iter" => 30,
    "save_start" => 0,
    "vol" => 0.2,
    "max_it" => 1000,
    "InitType" => ["Rand", "Line"],
    "InitFile" => "",
    "InitKey" => "",
    "scheme" => [
            SCHEME_NULL, 
            SCHEME_CORRECT, 
            SCHEME_BOUNDARY | SCHEME_CORRECT, 
            SCHEME_OLD,
            SCHEME_CHANGE,
            SCHEME_WALK,
            SCHEME_WINDOW,
            SCHEME_R_CORRECT
        ],
    "rand_rate" => 0.5,
    "rand_kernel_dim" => 4,

    # model parameter
    "N" => 600,  # 240 for Line initialization, 240 ÷ 2 ÷ 20
    "dim" => 2,
    "L" => 1.
];
comments = "debug"

run_with_configs(vec_configs, comments)
