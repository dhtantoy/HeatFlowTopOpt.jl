using HeatFlowTopOpt
using Logging 
using Gridap 

N = 480
prefix = "vtk_test/test_$N"
pvd = createpvd(prefix)
lg = ConsoleLogger()

vec_configs = [
    # pde parameter
    "β₁" => 0.,
    "β₂" => 0.1,
    "β₃" => 1,
    "δt" => 8e-3,
    "α⁻" => 417.5,
    "α₋" => 0.,
    "kf" => 0.1624,
    "ks" => 40.47,
    "Re" => 5988.,
    "γ" => 1027.6,
    "Ts" => 1.,
    "ud⋅n" => 0.,
    "Td" => 0.,
    "g⋅n" => 10.,
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
    "vol" => 0.4,
    "max_it" => 5,
    "InitType" => "Line",
    "is_correct" => true,
    "is_restart" => false,       # if not correction, then restart is off.
    "is_vol_constraint" => true, # if false, then set val to a scalar.
    "is_bdupdate" => false,

    # model parameter
    "N" => N,  # 240 for Line initialization, 240 ÷ 2 ÷ 20
    "dim" => 2,
    "L" => 1.
];

# debug
HeatFlowTopOpt.singlerun(Dict(vec_configs), prefix, pvd, lg; debug= true)
savepvd(pvd)

# ## run
# run_with_configs(vec_configs, "----") 


