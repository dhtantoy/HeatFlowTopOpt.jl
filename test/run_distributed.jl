using Distributed

# # run at 138
addprocs([
            ("c95", 8),
            ("c0130", 8),
            # ("yhxiang@197", 16),
            # ("c0123", 8),
            # ("c0124", 8),
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
    "β₁" => [0., 30.],
    "β₂" => 0.,
    "β₃" => 1,
    "δt" => 8e-3,
    "δu" => 5e-2,
    "α⁻" => 417.5,
    "kf" => 0.1624,
    "ks" => 40.47,
    "Re" => 5988.,
    "γ" => 1027.6,
    "Ts" => 1.,
    "ud⋅n" => 0.,
    "Td" => 0.0,
    "g⋅n" => [30., 100.],
    "Ts" => 1.,

    # motion paramter
    "up" => 0.95,
    "down" => 0.05,
    "τ₀" => 3e-4,
    "motion_tag" => "conv",

    # top opt parameter
    "correct_rate" => 0.5,
    "ϵ_ratio" => 0.5,
    "ϵ" => 10., 
    "save_iter" => 30,
    "save_start" => 0,
    "vol" => [0.3, 0.5],
    "max_it" => 1000,
    "InitType" => ["Rand", "Line"],
    "stable_scheme" => [STABLE_OLD, STABLE_RANDOM, STABLE_OLD | STABLE_RANDOM],
    "stable_rand_rate" => 0.6,
    "rand_kernel_dim" => 5,

    # model parameter
    "N" => 240,  # 240 for Line initialization, 240 ÷ 2 ÷ 20
    "dim" => 2,
    "L" => 1.
];
comments = "simpified, old/rand/old-rand, not commit"

run_with_configs(vec_configs, comments)
