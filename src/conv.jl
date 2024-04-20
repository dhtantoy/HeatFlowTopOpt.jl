
"""
convolution. Pd = (N - 1) >> 1 where N is the size of matrix accepted.
"""
struct Conv{T, Pd, D, Tc, Tp, Tpi}
    shift_gauss::Array{T}   # fft of gaussian kernel
    τ::Ref{T}               # 
    rfftA::Array{Tc, D}      # pre-allocate for rfft(A) and Rfft(A) .* shift_gauss
    irfftA::Array{T, D}      # pre-allocate for irfft(...)
    out::Array{T, D}        # pre-allocate for output result
    P_rfft::Tp              # plan for rfft
    P_irfft::Tpi            # plan for irfft
    Conv(T::Type, D::Int, N::Int, τ, nth::Int= Threads.nthreads(); time= 1) = begin
        @assert isodd(N) "Nx must be odd."
        @info "------------- convalution setting -------------"
        nth = min(nth, 4)
        @info @green "setting the number of FFTW threads to $nth..."
        FFTW.set_num_threads(nth)
        
        _sz = N - 1
        pdsz = _sz >> 1

        @info @green "generating convalution kernel..."
        shift_gauss = conv_kernel(T, _sz, τ, Val(D))
        
        rfftA = rfft(shift_gauss)
        irfftA = similar(shift_gauss)
        out = Array{T, D}(undef, ntuple(_->N, D)...)
        
        @info @green "planning for rfft..."
        P_rfft = plan_rfft(irfftA; flags= FFTW.PATIENT, timelimit= time)

        @info @green "planning for irfft..."
        P_irfft = plan_irfft(rfftA, size(shift_gauss, 1); flags= FFTW.PATIENT, timelimit= time)
        
        @info @green "done."
        @info "-------------------------------------------"
        new{T, pdsz, D, eltype(rfftA), typeof(P_rfft), typeof(P_irfft)}(shift_gauss, Ref(convert(T, τ)), rfftA, irfftA, out, P_rfft, P_irfft)
    end
end
get_pdsz(::Conv{T, Pd}) where {T, Pd} = 2Pd
get_kernel(c::Conv) = c.shift_gauss
get_kernel_updata_method(::Conv) = conv_kernel!

"""
Pd denotes padding size. Pd = (N - 1) >> 1 where N is the size of Matrix accepted.
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
generate convalution kernel.
"""
@generated function conv_kernel(::Type{T}, pdsz, τ, ::Val{D}) where {T, D}
    quote
        sz = 2pdsz
        A = @ncall $D Array{$T, $D} undef _ -> sz
        conv_kernel!(A, τ)
        A
    end
end
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
compute convolution of A, remenber to copy the c.out if another convalution need to be computed soon.
"""
@generated function (c::Conv{T1, Pd, D})(A::ExArray{T2, Pd, D}) where {T1,T2, Pd, D}
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
        shift_gauss = c.shift_gauss
        # rfftA = c.rfftA
        rfftA = c.P_rfft * A
        irfftA = c.irfftA
        out = c.out
        # mul!(rfftA, c.P_rfft, A)

        @inbounds @fastmath @nloops $D i rfftA begin
            (@nref $D rfftA i) = (@nref $D shift_gauss i) * (@nref $D rfftA i)
        end

        mul!(irfftA, c.P_irfft, rfftA)
        
        $(sub)
    end
end

function (c::Conv)(A)
    ExA = ExArray(A)
    c(ExA)
    nothing
end

function update_tau!(conv::Conv, ratio) 
    conv.τ[] *= ratio
    conv_kernel!(conv.shift_gauss, conv.τ[])
end