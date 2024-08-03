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
end


"""
    computediffset!(i_dec::Array, i_inc::Array, iA::AbstractArray, iB::AbstractArray, Φ::AbstractArray)
efficiently compute the difference set of elements of two vectors `iA` and `iB`,
which are sorted by a third vector `Φ`, i.e. `ϕ[ iA[j] ] ≤ ϕ[ iA[j+1] ]` and `ϕ[ iB[j] ] ≤ ϕ[ iB[j+1] ]` for all j.
`iA` - `iB` is stored in `i_dec` and `iB` - `iA` is stored in `i_inc`. 
"""
function computediffset!(i_dec, i_inc, iA, iB, Φ) where {T1, T2}
    na = length(iA)
    nb = length(iB)
    resize!(i_dec, 0)
    resize!(i_inc, 0) 
    i = j = refi = refj = 1

    @inbounds while i <= na && j <= nb
        iϕ = iA[i]
        jϕ = iB[j]
        Δϕi = Φ[iϕ]
        Δϕj = Φ[jϕ]
        if Δϕi < Δϕj
            push!(i_dec, iϕ)
            i += one(i)
            refi = i 
        elseif Δϕi > Δϕj
            push!(i_inc, jϕ)
            j += one(j)
            refj = j
        else    
            if !_check_from_until(refj, iB, iϕ, Φ, Δϕi)
                push!(i_dec, iϕ)
            end
            if !_check_from_until(refi, iA, jϕ, Φ, Δϕj)
                push!(i_inc, jϕ)
            end
            i += one(i)
            j += one(j)
            if i <= na && Φ[ iA[i] ] > Δϕi
                refj = j
            end
            if j <= nb && Φ[ iB[j] ] > Δϕj
                refi = i
            end
        end  
    end
    @inbounds for k = i:na
        push!(i_dec, iA[k])
    end
    @inbounds for k = j:nb
        push!(i_inc, iB[k])
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

"""
    domain2mp4(tb_path::AbstractString; subdirs::Union{Vector{String}, Vector{Int}, Nothing}= nothing, [load_tag = "domain/χ"])
convert the tensorboard data in `tb_path` to mp4 files and store it in the same directory.
"""
function domain2mp4(tb_path; subdirs::Union{Vector{String}, Vector{Int}, Nothing}= nothing, load_tag = "domain/χ")
    
    if isnothing(subdirs)
        subdirs = filter(x -> isdir(joinpath(tb_path, x)), readdir(tb_path))
    elseif subdirs isa Vector{Int}
        subdirs = map(i -> "run_$(i)", subdirs) 
    end
    
    encoder_options = (color_range=2, crf=0, preset="veryslow")

    for subdir in subdirs
        @info "----------------------------------------------"
        @info "$subdir converting..."
        dir = joinpath(tb_path, subdir)
        tb = TBReader(dir)
        hist = MVHistory()

        TensorBoardLogger.map_summaries(tb; tags= load_tag) do tag, iter, val
            push!(hist, Symbol(tag), iter, val)
        end
        img_list = hist.storage[Symbol(load_tag)].values

        file = dir*".mp4"
        infile = dir*"_.mp4"
        first_img = first(img_list)
        sz = map(x -> x ÷ 2 * 2, size(first_img))
        open_video_out(infile, eltype(first_img), sz, framerate=2, encoder_options=encoder_options) do writer
            for img in img_list
                write(writer, img)
            end
        end
        run(`ffmpeg -i $(infile) -r 24 $(file) -y`)
        rm(infile)
    end
    return nothing
end

"""
    parse_stable_scheme(s::Vector)
convert the stable scheme from a vector of `Unsigned` to a vector of `String`.
"""
parse_stable_scheme(s::Vector) = parse_stable_scheme.(s)

"""
    parse_stable_scheme(scheme::Unsigned)
convert the stable scheme from `Unsigned` to `String`.
"""
function parse_stable_scheme(scheme::Unsigned)
    ret = String[]
    all_schemes = [
        STABLE_OLD => "old",
        STABLE_CORRECT => "correct",
        STABLE_BOUNDARY => "boundary",
    ] 
    for (k, v) in all_schemes
        if !iszero(scheme & k)
            push!(ret, v)
        end
    end
    return join(ret, '_')
end

"""
    parse_random_scheme(s::Vector)
convert the random scheme from a vector of `Unsigned` to a vector of `String`.
"""
parse_random_scheme(s::Vector) = parse_random_scheme.(s)

"""
    parse_random_scheme(scheme::Unsigned)
convert the random scheme from `Unsigned` to `String`.
"""
function parse_random_scheme(scheme::Unsigned)
    ret = String[]
    all_schemes = [
        RANDOM_CHANGE => "change",
        RANDOM_WALK => "walk",
        RANDOM_WINDOW => "window",
        RANDOM_PROB => "probability",
    ] 
    for (k, v) in all_schemes
        if !iszero(scheme & k)
            push!(ret, v)
        end
    end
    return join(ret, '_')
end