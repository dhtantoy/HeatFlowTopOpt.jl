"""
    abstract type Motion end

Implementation of a smoother. the following interface should be implemented:

- `(m::Motion)(out::Array, A::AbstractArray)`: compute the smoother of `A` and store the result in `out`.

- `get_pdsz(m::Motion)`: return the padding size of the smoother.

- `get_tau(m::Motion)`: return the parameter `τ` of the smoother.

- `get_kernel(m::Motion)`: return the kernel of the smoother.

- `update_tau!(m::Motion, ratio)`: update the parameter `τ` of the smoother with the given `ratio` (τ *= ratio) and recompute the kernel if necessary.
"""
abstract type Motion end

(m::Motion)(out, arg) = error("method $(typeof(m)) for arguments $(typeof(arg)) is not defined.")
get_pdsz(m::Motion) = error("method get_pdsz for $(typeof(m)) is not defined.")
get_tau(m::Motion) = error("method get_tau for $(typeof(m)) is not defined.")
get_kernel(m::Motion) = error("method get_kernel for $(typeof(m)) is not defined.")
update_tau!(m::Motion, ratio) = error("method update_tau! for $(typeof(m)) is not defined.")

## -------------- convolution --------------
"""
    struct Conv{T, Pd, D, N, Tc, Tp, Tpi} <: Motion

A convolution smoother. The kernel is a Gaussian function with standard deviation `τ`. The kernel is computed in the frequency domain using FFTW.

# Constructor
`Conv(T::Type, D::Int, N::Int, τ, [nth::Int= Threads.nthreads()]; <keyword arguments>)`

# Arguments
- `T`: the element type of the input array.
- `D`: the number of dimensions of the input array.
- `N`: the size of the input array in each dimension.
- `τ`: the standard deviation of the Gaussian kernel.
- `nth`: the number of threads to use for the FFTW computation.
- `time`: the seconds limit for the FFTW planning.
"""
struct Conv{T, Pd, D, N, Tc, Tp, Tpi} <: Motion
    shift_gauss::Array{T, D}   # fft of gaussian kernel
    τ::Ref{T}               # 
    rfftA::Array{Tc, D}      # pre-allocate for rfft(A) and Rfft(A) .* shift_gauss
    irfftA::Array{T, D}      # pre-allocate for irfft(...)
    P_rfft::Tp              # plan for rfft
    P_irfft::Tpi            # plan for irfft
    Conv(T::Type, D::Int, N::Int, τ, nth::Int= Threads.nthreads(); time= 1) = begin
        @assert isodd(N) "Nx must be odd."
        nth = min(nth, 4)
        FFTW.set_num_threads(nth)
        _sz = N - 1
        pdsz = _sz >> 1
        shift_gauss = conv_kernel(T, _sz, τ, Val(D))
        rfftA = rfft(shift_gauss)
        irfftA = similar(shift_gauss)
        P_rfft = plan_rfft(irfftA; flags= FFTW.PATIENT, timelimit= time)
        P_irfft = plan_irfft(rfftA, size(shift_gauss, 1); flags= FFTW.PATIENT, timelimit= time)
       
        new{T, pdsz, D, N, eltype(rfftA), typeof(P_rfft), typeof(P_irfft)}(shift_gauss, Ref(convert(T, τ)), rfftA, irfftA, P_rfft, P_irfft)
    end
end

"""
    get_tau(c::Conv)
return the parameter `τ` of the convolution smoother.
"""
get_tau(c::Conv) = c.τ[]

"""
    get_pdsz(c::Conv{T, Pd})
return the padding size of the convolution smoother.
"""
get_pdsz(::Conv{T, Pd}) where {T, Pd} = 2Pd

"""
    get_kernel(c::Conv)
return the kernel of the convolution smoother.
"""
get_kernel(c::Conv) = c.shift_gauss

"""
    struct ExArray{T, Pd, D} <: AbstractArray{T, D}
extend Array of `D`D with padding size `Pd`, and type of data is `T`.
"""
struct ExArray{T, Pd, D} <: AbstractArray{T, D}
    A::Array{T, D}
    ExArray(A::Array{T, D}) where {T, D}= begin
        m = size(A, 1)
        Pd = (m - 1) >> 1
        @assert Pd > 0 "size of A should be greater than 2."
        new{T, Pd, D}(A)
    end
end

function Base.show(io::IO, ::Conv{T, Pd, D}) where {T, Pd, D}
    str = """
    self-defined convalution function which accepted `ExArray{T, $(Pd), $D}`.
    """
    print(io, str)
end
function Base.show(io::IO, ::ExArray{T, Pd, D}) where {T, Pd, D}
    str = """
    extend Array of $(D)D with padding size $(Pd), and type of data is $(T).
    """
    print(io, str)
end

"""
    conv_kernel(T::Type, pdsz, τ, D)
generate the kernel of the convolution smoother.
"""
@generated function conv_kernel(::Type{T}, pdsz, τ, ::Val{D}) where {T, D}
    quote
        sz = 2pdsz
        A = @ncall $D Array{$T, $D} undef _ -> sz
        conv_kernel!(A, τ)
        A
    end
end

"""
    conv_kernel!(A::Array{T, D}, τ)
generate the kernel of the convolution smoother and store it in `A`.
"""
@generated function conv_kernel!(A::Array{T, D}, τ) where {T, D}
    @fastmath quote
        m = size(A, 1)
        @assert iseven(m) "side length of A must be even."
        @assert (@nall $D d -> size(A, d) == m) "convalution kernel must be square but got $(size(A))."
        
        B = similar(A)
        pdsz = m >> 1
        li = -pdsz:pdsz-1
        invt =  τ / 4.
        @inbounds @nloops $D x B begin
            (@nref $D B x) = exp(-(@ncall $D Base.:+ d -> li[x_d] * li[x_d]) * invt)
        end
        fftshift!(A, B)
        nothing
    end
end

@generated function Base.size(A::ExArray{T, Pd, D}) where {T, Pd, D} 
    quote
        @ntuple $D d -> size(A.A, d) + 2Pd - 1
    end
end
Base.getindex(A::ExArray, I::Int...) = A[I]
@generated function Base.getindex(A::ExArray{T, Pd, D}, I::Union{CartesianIndex, NTuple{D, Int}}) where {T, Pd, D}
    @fastmath quote
        sub = A.A
        @nextract $D m d -> size(A.A, d)
        @nextract $D x d -> I[d] - Pd
        # note that @inbounds is needes.
        @inbounds ret = ifelse( (@nany $D d -> x_d <= 0) || (@nany $D d -> x_d > m_d), zero(T), (@nref $D sub x))
        ret
    end
end


"""
    (c::Conv{T, Pd, D, N})(out::Array{T, D}, A::Union{ExArray{T, Pd, D}, Array{T, D}})
compute the convolution of `A` and store the result in `out`. Remember to copy the c.out if another convalution need to be computed soon.
"""
@generated function (c::Conv{T1, Pd, D, N})(out::Array{T1, D}, A::ExArray{T2, Pd, D}) where {T1, T2, Pd, D, N}
    # used to optimized computation
    if D == 2
        sub = quote
            @tturbo for i_1 = axes(out, 1)
                for i_2 = axes(out, 2)
                    out[i_1, i_2] = irfftA[i_1 + $Pd, i_2+$Pd]
                end
            end
        end
    elseif D == 3
        sub = quote
            @tturbo for i_1 = axes(out, 1)
                for i_2 = axes(out, 2)
                    for i_3 = axes(out, 3)
                        out[i_1, i_2, i_3] = irfftA[i_1 + $Pd, i_2+$Pd, i_3+$Pd]
                    end
                end
            end
        end
    else
        sub = quote
            @nloops $D i out begin
                (@nref $D out i) = (@nref $D irfftA l->i_l + $Pd)
            end 
        end
    end
    
    quote
        @assert (@nall $D d -> size(out, d) == N) "length in each dimension of out must be $N."
        shift_gauss = c.shift_gauss
        # rfftA = c.rfftA
        rfftA = c.P_rfft * A
        irfftA = c.irfftA
        # mul!(rfftA, c.P_rfft, A)

        @inbounds @fastmath @nloops $D i rfftA begin
            (@nref $D rfftA i) = (@nref $D shift_gauss i) * (@nref $D rfftA i)
        end

        mul!(irfftA, c.P_irfft, rfftA)
        
        $(sub)
    end
end
function (c::Conv)(out::Array, A::Array)
    ExA = ExArray(A)
    c(out, ExA)
    nothing
end

"""
    update_tau!(conv::Conv, ratio)
update the parameter `τ` of the convolution smoother with the given `ratio` (τ *= ratio) and recompute the kernel if necessary.
"""
function update_tau!(conv::Conv, ratio)
    v = conv.τ[]
    conv.τ[] *= ratio
    if !isapprox(v, conv.τ[])
        conv_kernel!(conv.shift_gauss, conv.τ[])
    end
    return nothing
end


## -------------- median filter --------------


## -------------- identity (copy!) --------------
function update_tau!(::typeof(copy!), ratio)
    return nothing
end
get_tau(::typeof(copy!)) = 0.


## -------------- weighted average filter --------------
"""
    struct GaussianFilter{T, D} <: Motion

# Constructor
`GaussianFilter(T::Type, D::Int, N::Int, τ; <keyword arguments>)`

# Arguments
- `T`: the element type of the input array.
- `D`: the number of dimensions of the input array.
- `N`: the size of the input array in each dimension.
- `τ`: the standard deviation of the Gaussian kernel.
- `pdsz`: the padding size of the Gaussian kernel.
"""
struct GaussianFilter{T, D} <: Motion
    kernel::Array{T, D}
    origin::CartesianIndex{D}
    τ::Ref{T}
    N::Int
    GaussianFilter(T::Type, D::Int, N::Int, τ; pdsz::Int= 1) = begin
        k = gaussian_kernel(T, N, pdsz, τ)
        o = CartesianIndex(ntuple(_ -> pdsz, D)...)
        new{T, D}(k, o, τ, N)
    end
end

"""
    gaussian_kernel(T::Type, N::Int, pdsz::Int, τ)
generate the kernel of the Gaussian filter.
"""
function gaussian_kernel(T::Type, N::Int, pdsz::Int, τ)
    sz = 2pdsz + one(pdsz)
    A = Array{T}(undef, sz, sz)
    gaussian_kernel!(A, N, pdsz, τ)

    return A
end

"""
    gaussian_kernel!(A::Array, N::Int, pdsz::Int, τ)
generate the kernel of the Gaussian filter and store it in `A`.
"""
function gaussian_kernel!(A::Array, N::Int, pdsz::Int, τ)
    τ̂ = N^2 * τ
    li = -pdsz:pdsz
    for i = eachindex(li)
        xi = li[i]
        for j = eachindex(li)
            xj = li[j]
            A[i, j] = exp( -(xi^2 + xj^2)/τ̂ ) / (π * τ̂)
        end
    end
    A ./= sum(A)
    nothing
end
function Base.show(io::IO, ::GaussianFilter{T, D}) where {T, D}
    str = """
    self-defined Gaussian Filter.
    """
    print(io, str)
end

"""
    get_pdsz(gf::GaussianFilter)
return the padding size of the Gaussian filter.
"""
get_pdsz(gf::GaussianFilter) = size(gf.kernel, 1) ÷ 2

"""
    get_tau(gf::GaussianFilter)
get_tau(gf::GaussianFilter) = gf.τ[]
"""
get_tau(gf::GaussianFilter) = gf.τ[]

"""
    get_kernel(gf::GaussianFilter)
return the kernel of the Gaussian filter.
"""
get_kernel(gf::GaussianFilter) = gf.kernel

"""
    update_tau!(gf::GaussianFilter, ratio)
update the parameter `τ` of the Gaussian filter with the given `ratio` (τ *= ratio) and recompute the kernel if necessary.
"""
function update_tau!(gf::GaussianFilter, ratio) 
    v = gf.τ[]
    gf.τ[] *= ratio
    pdsz = get_pdsz(gf)
    if !isapprox(v, gf.τ[])
        gaussian_kernel!(gf.kernel, gf.N, pdsz, gf.τ[])
    end
    return nothing
end

"""
    (gf::GaussianFilter{T, D})(out::Array{T, D}, A::Array)
compute the Gaussian filter of `A` and store the result in `out`.
"""
function (gf::GaussianFilter{T, 2})(out::Array, A::Array) where T
    m, n = size(out)
    kernel = gf.kernel
    @tturbo for J1 = axes(out, 1)
        for J2 = axes(out, 2)
            temp = zero(T)
            for I1 = axes(kernel, 1)
                for I2 = axes(kernel, 2)
                    i = I1 + J1 - 2
                    j = I2 + J2 - 2
                    _i = ifelse(i < 1, 1, 0) + ifelse(1 <= i <= m, i, 0) + ifelse(i > m, m, 0)
                    _j = ifelse(j < 1, 1, 0) + ifelse(1 <= j <= n, j, 0) + ifelse(j > n, n, 0)
                    temp += A[_i, _j] * kernel[I1, I2]
                end
            end
            out[J1, J2] = temp
        end
    end
    return nothing
end
