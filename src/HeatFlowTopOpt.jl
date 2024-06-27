module HeatFlowTopOpt
# Write your package code here.

using Base.Cartesian
using Base.Threads
using Distributed

using Random
using LinearAlgebra
using JLD2
using TOML
using Logging
using Dates

using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.CellData
using Gridap.FESpaces
import GridapGmsh: gmsh

using LoopVectorization
using Pipe
using TensorBoardLogger
using FFTW
using FillArrays
using SparseArrays
using DataFrames
using ValueHistories
using VideoIO

# # always use OPENBLAS_NUM_THREADS=1 if your application is multithreaded while
# # using OpenBLAS. This is to avoid oversubscription of threads.
# BLAS.set_num_threads(1)

export run_with_configs
export domain2mp4


include("utils.jl")
include("motion.jl")
include("grid.jl")
include("fem.jl")
include("update.jl")
include("tb.jl")
end
