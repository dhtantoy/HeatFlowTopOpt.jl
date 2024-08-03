#######  division of two array with a given weight array. #######
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
more efficient than `setdiff` for vectors ordered by a second vector.
    note that the length of `i_pre_all` is `bgM` and `i_post_all` is `curM`, where `curM` ≤ `bgM`
"""

"""
    computediffset!(i_dec, i_inc, iA, iB, Φ)
efficiently compute the difference set of elements of a sorted vector `iA` and a given `iB`,
`iA` - `iB` is stored in `i_dec` and `iB` - `iA` is stored in `i_inc`. 
"""
function computediffset!(idx_decrease::SzVector{T1}, idx_increase::SzVector{T2}, idx_exist_sort_pre::AbstractVector{T1}, idx_pick_sort_post::AbstractVector{T2}, Φ::Array) where {T1, T2}
    na = length(idx_exist_sort_pre)
    nb = length(idx_pick_sort_post)
    resize!(idx_decrease, 0)
    resize!(idx_increase, 0) 
    i = j = refi = refj = 1

    while i <= na && j <= nb
        iϕ = idx_exist_sort_pre[i]
        jϕ = idx_pick_sort_post[j]
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
            if !_check_from_until(refj, idx_pick_sort_post, iϕ, Φ, Δϕi)
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
            if j <= nb && Φ[ idx_pick_sort_post[j] ] > Δϕj
                refi = i
            end
        end  
    end
    @inbounds for k = i:na
        push!(idx_decrease, idx_exist_sort_pre[k])
    end
    @inbounds for k = j:nb
        push!(idx_increase, idx_pick_sort_post[k])
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

function TensorBoardLogger.deserialize_tensor_summary(::TensorBoardLogger.tensorboard.var"Summary.Value")
    return nothing 
end
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


const STABLE_OLD = 0x0001
const STABLE_CORRECT = 0x0100
const STABLE_BOUNDARY = 0x1000

const RANDOM_CHANGE = 0x0001
const RANDOM_WALK = 0x0010
const RANDOM_WINDOW = 0x0100
const RANDOM_PROB = 0x1000

const SCHEME_NULL = 0x0000

parse_stable_scheme(s::Vector) = parse_stable_scheme.(s)
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

parse_random_scheme(s::Vector) = parse_random_scheme.(s)
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