using Distributed

# # run at 138
addprocs(
        [
            ("c95", 2),
            ("c0130",2),
            ("yhxiang@197", 2),
            ("c0123", 2),
            ("c0124", 2),
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

@everywhere using HeatFlowTopOpt
@everywhere using MKL

vec_configs = [
    # pde parameter
    "β₁" => 0.,
    "β₂" => 0.,
    "β₃" => 1.,
    "δt" => 0.,
    "α⁻" => [10, 1000., ],
    "kf" => 0.,
    "ks" => 10.,
    "Re" => [10, 100.],
    "Pr" => 6.7,
    "Ts" => 1.,
    "Vdₓ" => [0.1, 0.01, 100, 1000],
    "Td" => 0.0,
    "Pd" => 0.,
    "Ts" => 1.,

    # motion paramter
    "up" => 0.95,
    "down" => 0.05,
    "τ₀" => 1.5,
    "motion_type" => "gf",
    "pdsz" => 3,

    # top opt parameter
    "correct_rate" => 0.5,
    "ϵ_ratio" => 0.5,
    "ϵ" => 20., 
    "save_iter" => 200,
    "save_start" => 0,
    "vol" => 0.7,
    "max_it" => 200,
    "InitType" => ["Line", "Lines"],
    "InitFile" => "",
    "InitKey" => "",
    "scheme" => SCHEME_NULL,
    "rand_rate" => 0.5,
    "rand_kernel_dim" => 4,

    # model parameter
    "Nc" => 300,  # 240 for Line initialization, 240 ÷ 2 ÷ 20
    "ModelFile" => "lcr_model"
];
comments = "newmodel"

run_with_configs(vec_configs, comments)
