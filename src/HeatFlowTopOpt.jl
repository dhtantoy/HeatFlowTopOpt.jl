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
export SCHEME_OLD_PHI, SCHEME_OLD
export SCHEME_CORRECT, SCHEME_PROB_CORRECT, SCHEME_RAND_CORRECT, SCHEME_CORRECT_REV
export getmodel
# UInt16 at most 16 cases
const U16_UNIT = 0x0001

# ---- more details in codes.
# ICTM
const SCHEME_NULL = U16_UNIT >> 1
# addition of χ_k, χ_{k+1}
const SCHEME_OLD = U16_UNIT << 0
# prediction-correction
const SCHEME_CORRECT = U16_UNIT << 1
# restrict to boundary
const SCHEME_BOUNDARY = U16_UNIT << 2
# randomly change χ_k and χ_{k+1}
const SCHEME_CHANGE = U16_UNIT << 3
# random noise of χ_k
const SCHEME_WALK = U16_UNIT << 4
# randomly partly update
const SCHEME_WINDOW = U16_UNIT << 5
# random correction with Φ
const SCHEME_PROB_CORRECT = U16_UNIT << 6
# random correction.
const SCHEME_RAND_CORRECT = U16_UNIT << 7
# addition of Φ_k, Φ_{k+1}
const SCHEME_OLD_PHI = U16_UNIT << 8
# reversed prediction-correction
const SCHEME_CORRECT_REV = U16_UNIT << 9

const SCHEME_ALL_CORRECT = SCHEME_CORRECT | SCHEME_PROB_CORRECT | SCHEME_RAND_CORRECT | SCHEME_CORRECT_REV

const ALL_SCHEME_PAIRS = [
    SCHEME_OLD => "old",
    SCHEME_CORRECT => "correct",
    SCHEME_BOUNDARY => "boundary",
    SCHEME_CHANGE => "change",
    SCHEME_WALK => "walk",
    SCHEME_WINDOW => "window",
    SCHEME_PROB_CORRECT => "prob_correct",
    SCHEME_RAND_CORRECT => "rand_correct",
    SCHEME_OLD_PHI => "old_phi",
    SCHEME_CORRECT_REV => "correct_rev",
] 


include("utils.jl")
include("motion.jl")
include("fem.jl")
include("update.jl")
include("multidomain.jl")
include("tb.jl")
include("post.jl")
end
