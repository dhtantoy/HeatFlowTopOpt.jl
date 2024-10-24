using HeatFlowTopOpt
using Logging 
using Gridap 
using Debugger
using MKL

const lg = ConsoleLogger()

vec_configs = [
    # pde parameter
    "η" => 0.,
    "α⁻" => 1e9,
    "E" => 3e9,
    "θ" => 0.3,
    "Emin" => 1e-9,
    "ν" => 1e-3,
    "Vdₓ" => 1.,

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
    "vol" => 0.1,
    "max_it" => 10,
    "InitType" => "Rand",
    "InitFile" => "",
    "InitKey" => "",
    "scheme" => SCHEME_NULL,
    "rand_rate" => 0.5,
    "rand_kernel_dim" => 4,

    # model parameter
    "Nc" => 300 * 0.6,  #
    "ModelFile" => "ela_flow"
];

base_config, appended_config_arr = HeatFlowTopOpt.parse_vec_configs(vec_configs)
map(eachindex(appended_config_arr)) do i
    config = merge(base_config, Dict(appended_config_arr[i]...))
    prefix = "vtk_test/gf_"
    pvd = createpvd(prefix)

    HeatFlowTopOpt.singlerun(config, prefix, pvd, lg, 1; debug= true)
    savepvd(pvd)
end

