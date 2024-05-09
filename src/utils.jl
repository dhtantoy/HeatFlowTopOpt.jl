#######  division of two array with a given weight array. #######

"""
vector for efficiently pushing and resizing.
"""
struct SzVector{T} <: AbstractVector{T}
    sz::Base.RefValue{Int64}
    data::Vector{T}
    SzVector(::Type{T}, n_hint, n_cur) where {T} = begin
        v = Vector{T}(undef, n_hint)
        new{T}(Ref(n_cur), v)
    end
end

Base.length(v::SzVector) = v.sz[]
Base.size(v::SzVector) = (v.sz[], )
Base.getindex(v::SzVector, args...) = getindex(v.data, args...)
Base.getindex(v::SzVector, ::Colon) = getindex(v.data, Base.OneTo(v.sz[]))
Base.setindex!(v::SzVector, args...) = setindex!(v.data, args...)
Base.resize!(v::SzVector, n::Int64)  = begin
    r = length(v.data)
    if r < n
        resize!(v.data, n +  ceil(Int, r / 10))
    end
    v.sz[] = n
end


"""
more efficient than `setdiff` for vectors ordered by a second vector.
    note that the length of `i_pre_all` is `bgM` and `i_post_all` is `curM`, where `curM` ≤ `bgM`
"""
function computeAB!(idx_decrease::SzVector{T1}, idx_increase::SzVector{T2}, idx_exist_sort_pre::AbstractVector{T1}, idx_pick_post::AbstractVector{T2}, Φ::Array) where {T1, T2}
    na = length(idx_exist_sort_pre)
    nb = length(idx_pick_post)
    resize!(idx_decrease, 0)
    resize!(idx_increase, 0) 
    i = j = refi = refj = 1

    while i <= na && j <= nb
        iϕ = idx_exist_sort_pre[i]
        jϕ = idx_pick_post[j]
        Δϕi = Φ[iϕ]
        Δϕj = Φ[jϕ]
        if Δϕi < Δϕj
            push!(idx_decrease, iϕ)
            i += one(i)
            refi = i 
        elseif Δϕi > Δϕj
            push!(idx_increase, jϕ)
            j += one(j)
            refj = j
        else    
            if !_check_from_until(refj, idx_pick_post, iϕ, Φ, Δϕi)
                push!(idx_decrease, iϕ)
            end
            if !_check_from_until(refi, idx_exist_sort_pre, jϕ, Φ, Δϕj)
                push!(idx_increase, jϕ)
            end
            i += one(i)
            j += one(j)
            if i <= na && Φ[ idx_exist_sort_pre[i] ] > Δϕi
                refj = j
            end
            if j <= nb && Φ[ idx_pick_post[j] ] > Δϕj
                refi = i
            end
        end  
    end
    @inbounds for k = i:na
        push!(idx_decrease, idx_exist_sort_pre[k])
    end
    @inbounds for k = j:nb
        push!(idx_increase, idx_pick_post[k])
    end
end

"""
more efficient than `in` for vectors ordered just for some positions.
"""
function _check_from_until(start::Integer, v::AbstractVector{T}, x::T, Δϕ::AbstractArray{T2}, Δϕi::T2)::Bool where {T, T2}
    i = start
    while Δϕ[ v[i] ] == Δϕi
        if v[i] == x
            return true
        end
        i += one(i)
    end
    return false
end


function make_path(path::AbstractString, mode::UInt16)
    mkpath(path)
    chmod(path, mode)
end