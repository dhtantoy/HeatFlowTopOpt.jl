using HeatFlowTopOpt
using Logging 
using Gridap 
using Debugger
using MKL

const lg = ConsoleLogger()

vec_configs = [
    # pde parameter
    "β₁" => 0.,
    "β₂" => 0.,
    "β₃" => 1,
    "δt" => 0.,
    "α⁻" => 1000,
    "kf" => 0.,
    "ks" => 10,
    "Re" => 100,
    "Pr" => 6.7,
    "Ts" => 1.,
    "Vdₓ" => 1.,
    "Td" => 0.0,
    "Pd" => 0.,
    "Ts" => 1.,

    # motion paramter
    "up" => 0.95,
    "down" => 0.05,
    "τ₀" => 3e-4,
    "motion_type" => "conv",
    "pdsz" => 3,

    # top opt parameter
    "correct_rate" => 0.5,
    "ϵ_ratio" => 0.5,
    "ϵ" => 10., 
    "save_iter" => 1,
    "save_start" => 0,
    "vol" => 0.6,
    "max_it" => 10,
    "InitType" => "Line",
    "InitFile" => "",
    "InitKey" => "",
    "scheme" => SCHEME_CORRECT,
    "rand_rate" => 0.5,
    "rand_kernel_dim" => 4,

    # model parameter
    "Nc" => 300,  # 240 for Line initialization, 240 ÷ 2 ÷ 20
    "ModelFile" => "lcr_model"
];

base_config, appended_config_arr = HeatFlowTopOpt.parse_vec_configs(vec_configs)
map(eachindex(appended_config_arr)) do i
    config = merge(base_config, Dict(appended_config_arr[i]...))
    prefix = "vtk_test/gf_"
    pvd = createpvd(prefix)

    HeatFlowTopOpt.singlerun(config, prefix, pvd, lg, 1; debug= true)
    savepvd(pvd)
end

