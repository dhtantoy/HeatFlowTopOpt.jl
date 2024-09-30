using HeatFlowTopOpt
using Logging 
using Gridap 

const lg = ConsoleLogger()

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
    "τ₀" => 1e-4,
    "motion_type" => "conv",

    # top opt parameter
    "correct_rate" => 0.5,
    "ϵ_ratio" => 0.5,
    "ϵ" => 10., 
    "save_iter" => 5,
    "save_start" => 0,
    "vol" => 0.3,
    "max_it" => 5,
    "InitType" => "Rand",
    "InitFile" => "",
    "InitKey" => "",
    "scheme" => SCHEME_PROB_CORRECT,
    "rand_rate" => 0.5,
    "rand_kernel_dim" => 4,

    # model parameter
    "Nc" => 50,  # 240 for Line initialization, 240 ÷ 2 ÷ 20
    "dim" => 2,
    "L" => 1.
];

base_config, appended_config_arr = HeatFlowTopOpt.parse_vec_configs(vec_configs)
map(eachindex(appended_config_arr)) do i
    config = merge(base_config, Dict(appended_config_arr[i]...))
    Nc = config["Nc"]
    prefix = "vtk_test/simp_"
    pvd = createpvd(prefix)

    # debug
    HeatFlowTopOpt.singlerun(config, prefix, pvd, lg, 1; debug= true)
    savepvd(pvd)

end