using HeatFlowTopOpt
using Logging 
using Gridap 

const lg = ConsoleLogger()

vec_configs = [
    # pde parameter
    "β₁" => 0.,
    "β₂" => 1.,
    "β₃" => 1.,
    "δt" => 8e-3,
    "α⁻" => 417.5,
    "α₋" => 0.,
    "kf" => 0.1624,
    "ks" => 40.47,
    "Re" => 5988.,
    "γ" => [1027.6, 1e5],
    "Ts" => 1.,
    "ud⋅n" => 0.,
    "Td" => 0.,
    "g⋅n" => 1e-3,
    "Ts" => 1.,


    # motion paramter
    "up" => 0.95,
    "down" => 0.05,
    "τ₀" => 1e-4,
    "motion_tag" => "conv",

    # top opt parameter
    "correct_ratio" => 0.5,
    "ϵ_ratio" => 0.5,
    "ϵ" => 20., 
    "save_iter" => 1,
    "vol" => 0.3,
    "max_it" => 10,
    "InitType" => "Rand",
    "is_correct" => true,
    "is_restart" => false,       # if not correction, then restart is off.
    "is_vol_constraint" => true, # if false, then set val to a scalar.
    "is_bdupdate" => false,

    # model parameter
    "N" => 10,  # 240 for Line initialization, 240 ÷ 2 ÷ 20
    "dim" => 2,
    "L" => 1.,
];

base_config, appended_config_arr = HeatFlowTopOpt.parse_vec_configs(vec_configs)
map(eachindex(appended_config_arr)) do i
    config = merge(base_config, Dict(appended_config_arr[i]...))
    N = config["N"]
    prefix = "vtk_test/temp"
    pvd = createpvd(prefix)

    # debug
    HeatFlowTopOpt.singlerun(config, prefix, pvd, lg, 1; debug= true)
    savepvd(pvd)

end




