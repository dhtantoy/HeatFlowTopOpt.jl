"""
    struct SzVector{T} <: AbstractVector{T}
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

    return nothing
end
function lclip!(v::SzVector, n::Int64)
    @assert v.sz[] >= n "the number of elements to shift should be less than or equal to the current size."

    @inbounds for i = 1:v.sz[] - n
        v.data[i] = v.data[i + n]
    end
    v.sz[] -= n
    return nothing
end
function rclip!(v::SzVector, n::Int64)
    v.sz[] -= n
    return nothing
end



"""
    computediffset!(sort_i_dec::Array, sort_i_inc::Array, iA::AbstractArray, iB::AbstractArray, wv::AbstractArray)
efficiently compute the difference set of elements of two vectors `iA` and `iB`,
which are sorted by a third vector `wv`, i.e. `wv[ iA[j] ] ≤ wv[ iA[j+1] ]` and `wv[ iB[j] ] ≤ wv[ iB[j+1] ]` for all j.
`iA` - `iB` is stored in `i_sort_i_decdec` and `iB` - `iA` is stored in `sort_i_inc` in ascending order. 
"""
function computediffset!(sort_i_dec, sort_i_inc, iA, iB, wv)
    na = length(iA)
    nb = length(iB)
    resize!(sort_i_dec, 0)
    resize!(sort_i_inc, 0) 
    i = j = refi = refj = 1

    @inbounds while i <= na && j <= nb
        iϕ = iA[i]
        jϕ = iB[j]
        Δϕi = wv[iϕ]
        Δϕj = wv[jϕ]
        if Δϕi < Δϕj
            push!(sort_i_dec, iϕ)
            i += one(i)
            refi = i 
        elseif Δϕi > Δϕj
            push!(sort_i_inc, jϕ)
            j += one(j)
            refj = j
        else    
            if !_check_from_until(refj, iB, iϕ, wv, Δϕi)
                push!(sort_i_dec, iϕ)
            end
            if !_check_from_until(refi, iA, jϕ, wv, Δϕj)
                push!(sort_i_inc, jϕ)
            end
            i += one(i)
            j += one(j)
            if i <= na && wv[ iA[i] ] > Δϕi
                refj = j
            end
            if j <= nb && wv[ iB[j] ] > Δϕj
                refi = i
            end
        end  
    end

    # tail of vectors.
    @inbounds for k = i:na
        push!(sort_i_dec, iA[k])
    end
    @inbounds for k = j:nb
        push!(sort_i_inc, iB[k])
    end
end

"""
    _check_from_until(start::Ti, v::AbstractVector{Tv}, x::Tv, Δϕ::AbstractArray{Tp}, Δϕi::Tp)::Bool where {Ti <: Integer, Tv, Tp}
check if `x` is in `v` from `start` to the last `i`th element with `Δϕ[ v[i] ] == Δϕi`. More efficient than `∈` for vectors ordered by a 
second vector `Δϕ` in ascending order.
"""
function _check_from_until(start::Ti, v::AbstractVector{Tv}, x::Tv, Δϕ::AbstractArray{Tp}, Δϕi::Tp)::Bool where {Ti <: Integer, Tv, Tp}
    i = start
    @inbounds while Δϕ[ v[i] ] == Δϕi
        if v[i] == x
            return true
        end
        i += one(i)
    end
    return false
end

"""
    make_path(path::AbstractString, mode::UInt16)
make a directory with `path` and set the mode of the directory to `mode`.
"""
function make_path(path::AbstractString, mode::UInt16)
    mkpath(path)
    chmod(path, mode)
end

function TensorBoardLogger.deserialize_tensor_summary(::TensorBoardLogger.tensorboard.var"Summary.Value")
    return nothing 
end


function remove_bitmode(scheme::Unsigned, mod::Unsigned)
    return scheme & (~ mod)
end

"""
    scheme_to_str(s::Vector)
convert the stable scheme from a vector of `Unsigned` to a vector of `String`.
"""
scheme_to_str(s::Vector) = scheme_to_str.(s)

"""
    parse_scheme(scheme::Unsigned)
convert the scheme from `Unsigned` to `String`.
"""
function scheme_to_str(scheme::Unsigned)
    ret = String[]
    
    for (k, v) in ALL_SCHEME_PAIRS
        if !iszero(scheme & k)
            push!(ret, v)
        end
    end
    return join(ret, '-')
end

"""
    str_to_scheme(v::Vector)
convert the scheme from a vector of `String` to a vector of `Unsigned`.
"""
str_to_scheme(v::Vector) = str_to_scheme.(v)

"""
    str_to_scheme(str::String)
convert the scheme from `String` to `Unsigned`.
"""
function str_to_scheme(str::String)
    ret = zero(typeof(SCHEME_NULL))
    ss = split(str, '-')

    for (k, v) in ALL_SCHEME_PAIRS
        if v in ss
            ret |= k
        end
    end
    return ret
end

macro check_tau(val)
    esc( quote
        if isnan($val)
            $val = 0.
        elseif isinf($val)
            throw(DivideError())
        end
    end )
end


