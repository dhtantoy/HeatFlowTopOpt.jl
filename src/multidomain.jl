

struct PermArray{T, D, Tp} <: AbstractArray{T, D}
    P::Vector{Tp}   # permutation
    A::Vector{T}  # real data
    AP::Array{T, D} # permuted array AP = A[P]
end
function PermArray(A::Vector{T}, P::Vector{Tp}, dim) where {T, Tp}
    n::Int = length(P)^(1/dim)
    sz = repeat([n], dim)
    AP = zeros(T, sz...)
    return PermArray{T, dim, Tp}(P, A, AP)
end
function PermArray(A::Vector{T}, ::Nothing, dim) where {T}
    n::Int = length(A)^(1/dim)
    sz = repeat([n], dim)
    return reshape(A, sz...)
end

for op in (:size, :getindex, :copy, :setindex!, :sum)
    @eval Base.$op(v::PermArray, args...; kwargs...) = $op(v.AP, args...; kwargs...)
end
for op in (:eachindex, :zero, :zeros)
    @eval Base.$op(v::PermArray) = $op(v.AP)
end

for op in (:init_chi!, :post_interpolate!, :post_phi!, 
            :rand_post_phi!, :iterateχ!, :nonsym_correct!)
    @eval function $op(v::PermArray, args...; kwargs...) 
        $op(v.AP, args...; kwargs...)
        v.A[v.P] = v.AP
    end
    return nothing
end
function Base.copy!(v::PermArray, args...)
    copy!(v.AP, args...)
    v.A[v.P] = v.AP
    return nothing
end
for op in (:copy!, :copyto!)
    @eval Base.$op(s::AbstractArray{T, D}, v::PermArray{T, D}) where {T, D} = $op(s, v.AP)
end
function _compute_node_value!(v::PermArray, args...)
    _compute_node_value!(v.A, args...)
    v.AP[:] = v.A[v.P]
    return nothing
end

for conv in (:Conv, :GaussianFilter)
    @eval function (conv::$conv)(out, in::PermArray)
        conv(out, in.AP)
        return nothing
    end
    @eval function (conv::$conv)(out::PermArray, in)
        conv(out.AP, in)
        out.A[out.P] = out.AP
        return nothing
    end
    @eval function (conv::$conv)(out::PermArray, in::PermArray)
        conv(out, in.AP)
        return nothing
    end
end


function symmetry!(v::PermArray, a; kwargs...)
    copy!(v.AP, a)
    reverse!(a; kwargs...)
    @. v.AP = (v.AP + a) / 2
    v.A[v.P] = v.AP
    return nothing
end


# Base.setindex!(v::PermArray, args...) = setindex!(v.AP, args...)
function Base.show(io::IO, v::PermArray)
    show(io, v.AP)
end


function createGrid(::Val{2}, file, N_cell, L= 1.0)
    gmsh.initialize()
    gmsh.model.add("tmp")

    lc = 0.1

    gmsh.model.geo.addPoint(-1L, 1L, 0, lc, 1)
    gmsh.model.geo.addPoint(-1L, 2L, 0, lc, 2)
    gmsh.model.geo.addPoint(0, 2L, 0, lc, 3)
    gmsh.model.geo.addPoint(0, 3L, 0, lc, 4)
    gmsh.model.geo.addPoint(3L, 3L, 0, lc, 5)
    gmsh.model.geo.addPoint(3L, 2L, 0, lc, 6)
    gmsh.model.geo.addPoint(4L, 2L, 0, lc, 7)
    gmsh.model.geo.addPoint(4L, 1L, 0, lc, 8)
    gmsh.model.geo.addPoint(3L, 1L, 0, lc, 9)
    gmsh.model.geo.addPoint(3L, 0, 0, lc, 10)
    gmsh.model.geo.addPoint(0, 0, 0, lc, 11)
    gmsh.model.geo.addPoint(0, 1L, 0, lc, 12)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)
    gmsh.model.geo.addLine(7, 8, 7)
    gmsh.model.geo.addLine(8, 9, 8)
    gmsh.model.geo.addLine(9, 10, 9)
    gmsh.model.geo.addLine(10, 11, 10)
    gmsh.model.geo.addLine(11, 12, 11)
    gmsh.model.geo.addLine(12, 1, 12)
    gmsh.model.geo.addLine(3, 12, 13)
    gmsh.model.geo.addLine(6, 9, 14)

    gmsh.model.geo.addCurveLoop([1, 2, 13, 12], 15)
    gmsh.model.geo.addCurveLoop([3, 4, 5, 14, 9, 10, 11, -13], 16)
    gmsh.model.geo.addCurveLoop([-14, 6, 7, 8], 17)

    gmsh.model.geo.addPlaneSurface([15], 1)
    gmsh.model.geo.addPlaneSurface([16], 2)
    gmsh.model.geo.addPlaneSurface([17], 3)

    for i ∈ [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14]
        gmsh.model.geo.mesh.setTransfiniteCurve(i, N_cell + 1)
    end 

    for i ∈ [4, 10]
        gmsh.model.geo.mesh.setTransfiniteCurve(i, 3N_cell + 1)
    end
    gmsh.model.geo.mesh.setTransfiniteSurface(1)
    gmsh.model.geo.mesh.setTransfiniteSurface(2, "Left", [4, 5, 10, 11])
    gmsh.model.geo.mesh.setTransfiniteSurface(3)

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(2)

    gmsh.model.addPhysicalGroup(0, 1:12, -1, "wall")
    gmsh.model.addPhysicalGroup(1, [2, 3, 4, 5, 6, 8, 9, 10, 11, 12], -1, "wall") 
    gmsh.model.addPhysicalGroup(1, [1], -1, "inlet")
    gmsh.model.addPhysicalGroup(1, [7], -1, "outlet")
    gmsh.model.addPhysicalGroup(2, [2], -1, "motion_domain")
    gmsh.model.addPhysicalGroup(2, [1, 3], -1, "fixed_domain")
    gmsh.model.addPhysicalGroup(2, [1, 2, 3], -1, "domain")

    gmsh.write(file)
    gmsh.finalize()

    return nothing
end


function getmodel(file, Nc)
    @assert iszero(Nc % 3) "Nc should be a multiple of 3"
    path = joinpath("model", file*"_"*string(Nc)*".msh")
    if isfile(path)
        # @info "file exist, loading from file";
        m = DiscreteModelFromFile(path)
    else
        # @info "file does not exist, creating new file at $path"
        createGrid(Val(2), path, Nc ÷ 3)
        m = DiscreteModelFromFile(path)
    end

    perm = partialsortperm(
        get_node_coordinates(m),
        1:(Nc+1)^num_dims(m); 
        lt= (p, q) -> sort_axes_lt(p, q, 0.5/Nc)
    )

    return m, collect(perm)
end

"""
need to be debugged.
"""
function sort_axes_lt(p::VectorValue{D}, q::VectorValue{D}, tol::Real= 1e-4) where D
    if p[1] < - tol || p[1] > 3 + tol
        return false
    end
    if q[1] < - tol || q[1] > 3 + tol
        return true
    end

    err = p - q 
    idx = findlast(x->abs(x) >= tol, err.data)
    
    return isnothing(idx) || err[idx] < zero(eltype(err)) 
end

