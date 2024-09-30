using HeatFlowTopOpt
using Logging 
using Plots
using Gridap 
using Debugger
using HeatFlowTopOpt: 
    Conv, PermArray, init_chi!, post_interpolate!, 
    post_phi!, rand_post_phi!, iterateχ!, nonsym_correct!,
    init_single_space, smooth_funcs!, pde_solve!



config = Dict(
    # pde parameter
    "β₁" => 0.,
    "β₂" => 0.,
    "β₃" => 1,
    "δt" => 8e-3,
    "α⁻" => 417.5,
    "kf" => 0.,
    "ks" => 10,
    "Re" => 300,
    "Pr" => 6.7,
    "Ts" => 1.,
    "Vd" => VectorValue(0., 0.),
    "Td" => 0.0,
    "Pd" => 100.,
    "Ts" => 1.,

    # motion paramter
    "up" => 0.95,
    "down" => 0.05,
    "τ₀" => 1e-3,
    "motion_type" => "conv",

    # top opt parameter
    "correct_rate" => 0.5,
    "ϵ_ratio" => 0.5,
    "ϵ" => 10., 
    "save_iter" => 1,
    "save_start" => 0,
    "vol" => 0.3,
    "max_it" => 20,
    "InitType" => "Line",
    "InitFile" => "",
    "InitKey" => "",
    "scheme" => SCHEME_PROB_CORRECT,
    "rand_rate" => 0.5,
    "rand_kernel_dim" => 4,

    # model parameter
    "Nc" => 90,  # 240 for Line initialization, 240 ÷ 2 ÷ 20
    "ModelFile" => "lcr_model"
);

vtk_file_prefix = "vtk_test/simp_"
vtk_file_pvd = createpvd(vtk_file_prefix)
tb_lg = ConsoleLogger()
run_i = 1
debug= false

