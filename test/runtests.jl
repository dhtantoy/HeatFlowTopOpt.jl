using HeatFlowTopOpt
using Logging 
using Gridap 

const lg = ConsoleLogger()

vec_configs = [
    # pde parameter
    "β" => 15,
    "q1" => 1.,
    "q2" => 100.,
    "k1" => 10.,
    "k2" => 1.,
    "Td" => 0.,

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
    "max_it" => 10,
    "InitType" => "Line",
    "InitFile" => "",
    "InitKey" => "",
    "scheme" => SCHEME_CHANGE,
    "rand_rate" => 0.5,
    "rand_kernel_dim" => 4,

    # model parameter
    "N" => 600,  # 240 for Line initialization, 240 ÷ 2 ÷ 20
    "dim" => 2,
    "L" => 1.
];

base_config, appended_config_arr = HeatFlowTopOpt.parse_vec_configs(vec_configs)
map(eachindex(appended_config_arr)) do i
    config = merge(base_config, Dict(appended_config_arr[i]...))
    N = config["N"]
    prefix = "vtk_test/simp_"
    pvd = createpvd(prefix)

    # debug
    HeatFlowTopOpt.singlerun(config, prefix, pvd, lg, 1; debug= true)
    savepvd(pvd)

end