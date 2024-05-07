using HeatFlowTopOpt
using Logging

lg = ConsoleLogger()

vec_configs =[
    # pde parameter
    "β₁" => 0.,
    "β₂" => 1.,
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
    "Td" => 0.0,
    "g⋅n" => 33.5,
    "Ts" => 1.,

    # motion paramter
    "up" => 0.95,
    "down" => 0.05,
    "τ₀" => 1e-3,
    "motion_tag" => "conv",

    # top opt parameter
    "correct_ratio" => 0.5,
    "ϵ_ratio" => 0.5,
    "ϵ" => 20., 
    "save_iter" => 5,
    "vol" => 0.4,
    "max_it" => 1000,
    "InitType" => "Line",
    "is_correct" => true,
    "is_restart" => true,
    "is_vol_constraint" => false, # if false, then set val to a scalar.
    "is_bdupdate" => true,

    # model parameter
    "N" => 240, # cell
    "dim" => 2,
    "L" => 1.
];

## debug
# HeatFlowTopOpt.singlerun(Dict(vec_configs), "vtk_test", lg)

## run
run_with_configs(vec_configs, "----") 


