using Pkg
Pkg.activate(".")

using HeatFlowTopOpt

configs =[
    "β₁" => 0.,
    "β₂" => 5e-2,
    "β₃" => 1,
    "correct_ratio" => 0.5,
    "ϵ_ratio" => 0.5,
    "ϵ" => 10., 
    "δt" => 8e-3,
    "up" => 0.95,
    "down" => 0.05,
    "save_iter" => 30,
    "N" => 254,
    "τ₀" => 1e-3,
    "α⁻" => 417.5,
    "Re" => 1000.,
    "ud⋅n" => 0.,
    "Td" => 0.0,
    "g⋅n" => 33.5,
    "Ts" => 1.,
    "vol" => 0.4,
];


run_with_configs(configs; it= 1000, debug= false)
