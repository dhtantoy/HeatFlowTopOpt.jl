"""
    random_window!(A, kernel, vol, i)
in-place generation of a random `0-1` `kernel` with seed `i` and volume `vol`,
and upsample and store it to the size of `A` as a window.
"""
function random_window!(A, kernel, ratio, seed) 
    rand!(Random.seed!(seed), kernel)
    T = eltype(A)

    @turbo for j = eachindex(kernel)
        kernel[j] = ifelse(kernel[j] <= ratio, one(T), zero(T))
    end

    m = size(kernel, 1)
    n = size(A, 1)
    a, b = divrem(n, m) 
    I = ones(eltype(kernel), a, a)
    c = b ÷ 2
    kron!(view(A, c+1:c+a*m, c+1:c+a*m), kernel, I)

    return nothing
end

"""
    post_interpolate!(A, B, w)
in-place update of `A` with `B` and `w` as the weight. `(1-w)A + wB -> A`, i.e.
`A + w(B - A) -> A`.
"""
function post_interpolate!(A, B, w::Real)
    r = 1 - w
    @turbo for i = eachindex(A)
        A[i] = B[i] * w + A[i] * r
    end
    return nothing
end
    
"""
    post_interpolate!(A, B, W)
in-place update of `A` with `B` and `W` as the weight. `(1-W)⊗A + W⊗B -> A`, i.e. 
`A + W⊗(B - A) -> A`.
"""
function post_interpolate!(A, B, W::AbstractArray{T}) where T
    @turbo for i = eachindex(A)
        A[i] = B[i] + W[i] * (A[i] - B[i])
        # A[i] = B[i] * W[i] + A[i] * (one(T) - W[i])
    end
    return nothing
end

"""
    post_phi!(Φ, Gτχ, Φ_max, Φ_min, [down, up])
in-place post processing of `Φ` with `Gτχ` and `down` and `up` as the threshold.
"""
function post_phi!(Φ::AbstractArray{T1}, Gτχ::AbstractArray{T2}, Φ_max::T1, Φ_min::T1, down::T2= typemin(T2), up::T2= typemax(T2)) where {T1, T2}
    Φ_max += one(T1)
    Φ_min -= one(T1)
    @inbounds for i = eachindex(Φ)
        if Gτχ[i] < down 
            Φ[i] = Φ_max
        elseif Gτχ[i] > up
            Φ[i] = Φ_min
        end
    end
    return nothing
end

"""
    rand_post_phi!(Φ, perm, Φ_max, n, seed)
in-place post porcessing, setting `n` random elements in `Φ` to `typemax(T)`.
"""
function rand_post_phi!(Φ::AbstractArray{T}, perm::AbstractArray{<:Integer}, Φ_max::T, n, seed) where {T}
    randperm!(Random.seed!(seed), perm)
    Φ_max += one(T)
    @turbo for j = 1:n
        Φ[ perm[j] ] = Φ_max
    end

    return nothing
end


"""
    iterateχ!(χ, order, Φ, M)
in-place iteration of `χ` with volume `M`, and the ascending order of selected `Φ` is stored in `order`.
"""
function iterateχ!(χ, order::SzVector{Int}, Φ::AbstractArray{T}, M) where {T}
    fill!(χ, zero(T))
    ix = order.data

    sortperm!(ix, vec(Φ); alg=PartialQuickSort( Base.oneto(M + 1) ))
    curM = M
    val = Φ[ ix[M+1] ]

    @inbounds while curM > 1 && Φ[ ix[ curM ] ] ≈ val
        curM -= 1
    end

    curM = curM > 1 ? curM : M

    @inbounds for i = Base.oneto(curM)
        χ[ ix[i] ] = one(T)
    end

    resize!(order, curM)
    nothing
end

"""
    get_sorted_idx!(order_idx, cache_val, cache_idx, χ, weight)
in-place computation of ascendingly ordered indices of selected `weight` (`χ` = 1).
"""
function get_sorted_idx!(order_idx::AbstractVector, cache_val::SzVector, cache_idx::SzVector, χ, weight)
    resize!(cache_val, 0)
    resize!(cache_idx, 0)
    T = eltype(χ)
    @inbounds for i = eachindex(χ)
        indicator = χ[i]
        if indicator == one(T)
            push!(cache_val, weight[i])
            push!(cache_idx, i)
        end
    end
    resize!(order_idx, length(cache_val))
    sortperm!(order_idx, cache_val; alg=QuickSort)

    # change `perm` to real index
    @inbounds for i = eachindex(order_idx)
        j = order_idx[i]
        order_idx[i] = cache_idx.data[ j ]
    end

    return nothing
end


"""
    phi_to_prob!(P, Φ, Φ_max)
in-place computation of probability (rescaled `Φ`) stored in `P`, and where Φ is smaller
has larger probability.
"""
function phi_to_prob!(P, Φ::AbstractArray{T}, Φ_max::T) where {T}
    Φ_max += one(T)     # avoid zero probability
    d = 1. / (length(Φ) * Φ_max - sum(Φ))
    @turbo for i = eachindex(Φ)
        P[i] = d * (Φ_max - Φ[i])
    end
    
    return nothing
end

"""
    prob_to_weight!(weight, P, cache_val)
in-place computation of weight stored in `weight` with probability `P`, and 
where P is larger has smaller `weight`.
"""
function prob_to_weight!(weight, P, cache_idx::AbstractArray{T}) where {T <: Integer}
    a = one(T):length(P)
    wv = pweights(P)
    sample!(a, wv, cache_idx; replace= false)
    
    @inbounds for i = eachindex(cache_idx)
        weight[ cache_idx[i] ] = i
    end

    return nothing
end


"""
    nonsym_correct!(χ, sort_idx_dec, sort_idx_inc, θ)
in-place correction of `χ` with rate `θ` and update `sort_idx_dec` and `sort_idx_inc`, 
where `sort_idx_dec` (1->0) and `sort_idx_inc` (0->1) are ascendingly sorted indices.
"""
function nonsym_correct!(χ, sort_idx_dec, sort_idx_inc, θ)
    na = length(sort_idx_dec)
    nb = length(sort_idx_inc)
    @assert 0 < θ < 1 "θ should be in (0, 1)."
    @assert na > 0 || nb > 0 "na or nb should be positive."

    ma = ceil(Int, θ*na)
    mb = ceil(Int, θ*nb)
    T = eltype(χ)

    @inbounds for i = 1:ma
        j = sort_idx_dec[i]
        χ[j] = one(T)
    end
    @inbounds for i = nb - mb + 1:nb
        j = sort_idx_inc[i]
        χ[j] = zero(T)
    end

    lclip!(sort_idx_dec, ma)
    rclip!(sort_idx_inc, mb)

    return nothing
end



