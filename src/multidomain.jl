

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
    @eval function $op(x, v::PermArray, args...; kwargs...) 
        $op(x, v.AP, args...; kwargs...)
        return nothing
    end
    @eval function $op(x::PermArray, v::PermArray, args...; kwargs...) 
        $op(x, v.AP, args...; kwargs...)
        return nothing
    end
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

phi_to_prob!(P::Array, W::PermArray, x) =  phi_to_prob!(P, W.AP, x)
   

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


function createGrid(::Val{2}, file, N_cell, L= 1e-4)
    gmsh.initialize()
    gmsh.model.add("tmp")

    lc = 0.1
    lm = 0.6e-4
    gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
    gmsh.model.geo.addPoint(4L, 0, 0, lc, 2)
    gmsh.model.geo.addPoint(4L, L, 0, lc, 3)
    gmsh.model.geo.addPoint(0, L, 0, lc, 4)
    # inner points
    gmsh.model.geo.addPoint((4L - lm) * 0.5, 0, 0, lc, 5)
    gmsh.model.geo.addPoint((4L + lm) * 0.5, 0, 0, lc, 6)
    gmsh.model.geo.addPoint((4L + lm) * 0.5, lm, 0, lc, 7)
    gmsh.model.geo.addPoint((4L - lm) * 0.5, lm, 0, lc, 8)
    
    
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    # inner lines
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)
    gmsh.model.geo.addLine(7, 8, 7)
    gmsh.model.geo.addLine(8, 5, 8)
    
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 9)
    gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 10)
    
    gmsh.model.geo.addPlaneSurface([9], 1)
    gmsh.model.geo.addPlaneSurface([10], 2)
    
    gmsh.model.geo.mesh.setTransfiniteCurve(1, 4N_cell + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(3, 4N_cell + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(2, N_cell + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(4, N_cell + 1)
    
    for i ∈ [5, 6, 7, 8]
        gmsh.model.geo.mesh.setTransfiniteCurve(i, 0.6 * N_cell + 1)
    end 
    
    gmsh.model.geo.mesh.setTransfiniteSurface(1)
    gmsh.model.geo.mesh.setTransfiniteSurface(2)
    
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(2, [2], 2, 1)
    
    gmsh.model.mesh.generate(2)
    
    gmsh.model.addPhysicalGroup(0, [1,2,3,4], -1, "wall")
    gmsh.model.addPhysicalGroup(1, [1, 3], -1, "wall")
    gmsh.model.addPhysicalGroup(1, [2], -1, "outlet")
    gmsh.model.addPhysicalGroup(1, [4], -1, "inlet")
    gmsh.model.addPhysicalGroup(2, [2], -1, "motion_domain")
    gmsh.model.addPhysicalGroup(2, [1, 2], -1, "domain")
    
    # gmsh.fltk.run()
    gmsh.write(file)
    gmsh.finalize()

    return nothing
end


function getmodel(file, Nc)
    path = joinpath("model", file*"_"*string(Nc)*".msh")
    if isfile(path)
        # @info "file exist, loading from file";
        m = DiscreteModelFromFile(path)
    else
        # @info "file does not exist, creating new file at $path"
        createGrid(Val(2), path, Nc / 0.6)
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
function sort_axes_lt(p::VectorValue{D}, q::VectorValue{D}, tol::Real= 1e-7) where D
    if p[1] < 1.7e-4 - tol || p[1] > 2.3e-4 + tol || p[2] > 0.6e-4 + tol
        return false
    end
    if q[1] < 1.7e-4 - tol || q[1] > 2.3e-4 + tol || q[2] > 0.6e-4 + tol
        return true
    end

    err = p - q 
    idx = findlast(x->abs(x) >= tol, err.data)
    
    return isnothing(idx) || err[idx] < zero(eltype(err)) 
end

