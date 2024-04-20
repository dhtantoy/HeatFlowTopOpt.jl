module HeatFlowTopOpt
# Write your package code here.

using Base.Cartesian
using Base.Threads

using Random
using LinearAlgebra
using JLD2
using Logging
using Dates

using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.CellData
using Gridap.FESpaces

using LoopVectorization
using Plots
using Pipe
using TensorBoardLogger
using FFTW
using FillArrays
using Term
using DataFrames
using Sockets

# export Conv, update_tau!
# export initmodel, femsolve!, Phi!, initspace
# export singlerun

export run_with_configs


include("utils.jl")
include("conv.jl")
include("fem.jl")
include("update.jl")
include("tb.jl")
end
