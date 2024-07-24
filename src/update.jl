function random_chi!(cache_arr_rand_χ::Array{T}, vol, i) where T
    rand!(Random.seed!(i), cache_arr_rand_χ)

    @turbo for i = eachindex(cache_arr_rand_χ)
        cache_arr_rand_χ[i] = ifelse(cache_arr_rand_χ[i] <= vol, one(T), zero(T))
    end

    return nothing
end

function post_chi!(cache_arr_χ, another)
    @turbo for i = eachindex(cache_arr_χ)
        cache_arr_χ[i] = (another[i] + cache_arr_χ[i]) * 0.5
    end
    return nothing
end

function post_phi!(cache_Φ::Array{T1}, cache_Gτχ::Array{T2}, down::T2= typemin(T2), up::T2= typemax(T2)) where {T1, T2}
    @inbounds for i = eachindex(cache_Φ)
        if cache_Gτχ[i] < down 
            cache_Φ[i] = typemax(T1)
        elseif cache_Gτχ[i] > up
            cache_Φ[i] = typemin(T1)
        end
    end
    return nothing
end

function iterateχ!(cache_χ, cache_Φ::Array{T}, idx_pick_post::SzVector{Int}, M) where {T}
    @turbo cache_χ .= zero(T)

    ix = idx_pick_post.data

    sortperm!(ix, vec(cache_Φ); alg=PartialQuickSort( Base.oneto(M) ))
    curM = M 

    val = cache_Φ[ ix[M] ]

    while curM > 1 && cache_Φ[ ix[ curM ] ] ≈ val
        curM -= 1
    end

    @inbounds for i = Base.oneto(curM)
        cache_χ[ ix[i] ] = one(T)
    end

    resize!(idx_pick_post, curM)
    nothing
end


"""
get the indices where χ=1 and then order it with value Φ
"""
function get_ordered_idx!(idx_exist_sort::AbstractVector{Int}, cache_val_exist::SzVector{T2}, cache_idx_exist::SzVector{T1}, χ::Array{T1}, Φ::Array{T2}) where {T1, T2}
    resize!(cache_val_exist, 0)
    resize!(cache_idx_exist, 0)
    @inbounds for i = eachindex(χ)
        indicator = χ[i]
        if indicator == 1
            push!(cache_val_exist, Φ[i])
            push!(cache_idx_exist, i)
        end
    end
    resize!(idx_exist_sort, length(cache_val_exist))
    
    sortperm!(idx_exist_sort, cache_val_exist; alg=QuickSort)

    # change `perm` to real index
    @inbounds @threads :dynamic for i = eachindex(idx_exist_sort)
        j = idx_exist_sort[i]
        idx_exist_sort[i] = cache_idx_exist.data[ j ]
    end
end


"""

"""
function correct!(χ::Array{T}, θ, idx_decrease::AbstractVector, idx_increase::AbstractVector, Φ::Array) where T
    na = length(idx_decrease)
    nb = length(idx_increase)
    # @assert na != 0 && nb != 0 "length of `ia` and `ib` should not be equal to zero but ($(na), $(nb))"
    # compute real number of points to correct.
    ma = ceil(Int, θ*na)
    mb = ceil(Int, θ*nb)

    if ma > 0
        δ₁ = Φ[ idx_decrease[ma] ]
        @inbounds @threads for i = eachindex(idx_decrease)
            j = idx_decrease[i]
            χ[j] = ifelse(Φ[j] <= δ₁, one(T), zero(T))
        end
    end
    if mb > 0
        δ₂ = Φ[ idx_increase[ nb - mb + 1] ]
        @inbounds @threads for i = eachindex(idx_increase)
            j = idx_increase[i]
            χ[j] = ifelse(Φ[j] >= δ₂, zero(T), one(T))
        end
    end
end

