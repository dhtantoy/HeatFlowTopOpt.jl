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
export STABLE_BOUDARY, STABLE_CORRECT, STABLE_OLD, STABLE_RANDOM


# stablization scheme 
const STABLE_OLD = 0x0001
const STABLE_RANDOM = 0x0010
const STABLE_CORRECT = 0x0100
const STABLE_BOUDARY = 0x1000


include("utils.jl")
include("motion.jl")
include("fem.jl")
include("update.jl")
include("tb.jl")
end
