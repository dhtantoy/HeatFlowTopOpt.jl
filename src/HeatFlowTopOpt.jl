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
using StatsBase: pweights, sample!

# # always use OPENBLAS_NUM_THREADS=1 if your application is multithreaded while
# # using OpenBLAS. This is to avoid oversubscription of threads.
# BLAS.set_num_threads(1)

export run_with_configs
export domain2mp4
export STABLE_BOUNDARY, STABLE_CORRECT, STABLE_OLD
export RANDOM_WALK, RANDOM_CHANGE, RANDOM_WINDOW, RANDOM_PROB
export SCHEME_NULL

const STABLE_OLD = 0x0001
const STABLE_CORRECT = 0x0100
const STABLE_BOUNDARY = 0x1000

const RANDOM_CHANGE = 0x0001
const RANDOM_WALK = 0x0010
const RANDOM_WINDOW = 0x0100
const RANDOM_PROB = 0x1000

const SCHEME_NULL = 0x0000

include("utils.jl")
include("motion.jl")
include("fem.jl")
include("update.jl")
include("tb.jl")
end
