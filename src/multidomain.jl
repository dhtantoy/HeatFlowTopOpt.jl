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


