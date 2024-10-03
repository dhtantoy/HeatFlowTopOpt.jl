module HeatFlowTopOpt
# Write your package code here.

using Base.Cartesian
using Base.Threads
using Distributed

using Random
using LinearAlgebra
using JLD2
import TOML
using Logging
using Dates: now, format as dformat, DateTime

using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.CellData
using Gridap.FESpaces
using BubbleSpaces
import GridapGmsh: gmsh

using LoopVectorization
using TensorBoardLogger
import ProtoBuf as PB
using FFTW
using FillArrays: Fill
using SparseArrays: sparse
using DataFrames
using ValueHistories: MVHistory
using VideoIO: open_video_out
using StatsBase: pweights, sample!
using FileIO: save
using Plots
using Printf: Format, @sprintf


# # always use OPENBLAS_NUM_THREADS=1 if your application is multithreaded while
# # using OpenBLAS. This is to avoid oversubscription of threads.
# BLAS.set_num_threads(1)

export run_with_configs
export domain2mp4
export SCHEME_NULL
export SCHEME_BOUNDARY, SCHEME_WALK, SCHEME_CHANGE, SCHEME_WINDOW
export SCHEME_OLD_PHI, SCHEME_OLD, SCHEME_GRADIENT
export SCHEME_CORRECT, SCHEME_PROB_CORRECT, SCHEME_RAND_CORRECT, SCHEME_CORRECT_REV
export getmodel


include("utils.jl")
include("motion.jl")
include("fem.jl")
include("update.jl")
include("multidomain.jl")
include("tb.jl")
include("post.jl")
end
