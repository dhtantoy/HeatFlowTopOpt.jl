struct NullTriangulation{Dc, Dp} <: Triangulation{Dc, Dp} 
    NullTriangulation(model::DiscreteModel) = begin
        Dp = num_point_dims(model)
        Dc = num_cell_dims(model)
        new{Dc, Dp}()
    end
end

struct NullFESpace{T} <: FESpace 
    trian::T
end

struct NullFEFunction{T, DT} <: FEFunction 
    space::T
    domain_style::DT
end


Gridap.get_triangulation(ns::NullFESpace) = ns.trian
Gridap.num_free_dofs(::NullFESpace) = 0
Gridap.num_cells(::NullTriangulation) = 0
Gridap.get_triangulation(nf::NullFEFunction) = get_triangulation(nf.space)
Gridap.DomainStyle(nf::NullFEFunction) = nf.domain_style

function Gridap.Triangulation(m::DiscreteModel, domain_tags::Vector)
    return isempty(domain_tags) ? NullTriangulation(m) : Triangulation(m, tags= domain_tags)
end
function Gridap.FESpace(nt::NullTriangulation, arg...; kwarg...)
    return NullFESpace(nt)
end

function Gridap.FEFunction(ns::NullFESpace, arg...; kwarg...)
    return NullFEFunction(ns, ReferenceDomain())
end

Gridap.:+(a::FEFunction, ::NullFEFunction) = a
Gridap.:+(::NullFEFunction, a::FEFunction) = a

Gridap.:*(a::NullFEFunction, ::Number) = a
Gridap.:*(::Number, a::NullFEFunction) = a

"""
return
    spaces:                                                     \\
        - motion_space: fe space based on computational domain; \\
        - fixed_space: fe space based on the rest domain;       \\
    trians:                                                     \\
        - Ω: Triangulation of model;                            \\
        - Γs: BoundaryTriangulation of model;                   \\
    measures:                                                   \\
        - dx: Measure on Ω;                                     \\
        - dΓs: Measures on Γs;                                  \\
    perm: permutation map generated from `sortperm(aux_Ω)`;     \\
"""
function initmodel(model::DiscreteModel, motion_domain_tags::Vector, fixed_domain_tags::Vector, boundary_tags::Vector)
    @info "generating triangulation ..."
    Ω = Triangulation(model)
    motion_Ω = Triangulation(model, motion_domain_tags)
    fixed_Ω = Triangulation(model, fixed_domain_tags)
    Γs = map(boundary_tags) do tags 
        BoundaryTriangulation(model; tags= tags)
    end 

    @info "generating measure ..."
    dx = Measure(Ω, 4)
    dΓs = map(Γs) do Γ
        Measure(Γ, 4)
    end

    @info "computing permutation map ..."
    perm = sortperm(get_node_coordinates(motion_Ω); lt= sort_axes_lt)

    @info "constrcting fe space for motion and fixed domain..."
    motion_space = TestFESpace(motion_Ω, ReferenceFE(lagrangian, Float64, 1); conformity= :H1)
    fixed_space = TestFESpace(fixed_Ω, ReferenceFE(lagrangian, Float64, 1); conformity= :H1)
 
    return (motion_space, fixed_space), (Ω, Γs), (dx, dΓs), perm
end

"""
return 
    motion_cache_fe_funcs:                                      \\
        - motion_cache_fe_χ: finite element function χ;         \\
        - motion_cache_fe_Gτχ: finite element function Gτχ;     \\
        - motion_cache_fe_α: finite element function of coefficient α;  \\
        - motion_cache_fe_κ: finite element function of coefficient κ; 
"""
function initfefuncs(motion_space)
    @info "constructing fefunction fe_χ, fe_Gτχ fe_α, fe_κ ..."
    motion_cache_fe_χ = FEFunction(motion_space, zeros(Float64, num_free_dofs(motion_space)))
    motion_cache_fe_Gτχ = FEFunction(motion_space, zeros(Float64, num_free_dofs(motion_space)))
    motion_cache_fe_α = FEFunction(motion_space, zeros(Float64, num_free_dofs(motion_space)))
    motion_cache_fe_κ = FEFunction(motion_space, zeros(Float64, num_free_dofs(motion_space)))

    return motion_cache_fe_χ, motion_cache_fe_Gτχ, motion_cache_fe_α, motion_cache_fe_κ
end


"""
return
    motion_cache_arr_χ: initialiation of χ in the form of array    \\
        - All \\
        - Line \\
        - Net \\
        - Rand \\
    motion_cache_arr_χ₂: cache of χ₂ in the form of array;          \\
    motion_cache_arr_Gτχ: cache of Gτχ in the form of array;        \\
    motion_cache_arr_Gτχ₂: cache of Gτχ₂ int the form of array.     
"""
function initcachechis(InitType, motion_sapce; vol= 0.4, seed= 1)
    dim = get_triangulation(motion_sapce) |> num_point_dims
    np = num_free_dofs(motion_sapce)
    N_node::Int = np ^ (1//dim)

    @info "------------- generate χ₀ -------------"
    if InitType == "All"
        motion_cache_arr_χ = ones(Float64, repeat([N_node], dim)...)
    else
        motion_cache_arr_χ = zeros(Float64, repeat([N_node], dim)...)

        n = 20
        p = Iterators.partition(1:(N_node >> 1), N_node ÷ n) |> collect

        if InitType == "Net"
            for I in p[2:2:end]
                motion_cache_arr_χ[:, I] .= 1
                motion_cache_arr_χ[:, end .- I] .= 1
                motion_cache_arr_χ[I, :] .= 1
                motion_cache_arr_χ[end .- I, :] .= 1
            end
        elseif InitType == "Line"
            for I in p[2:2:end]
                motion_cache_arr_χ[:, I] .= 1
                motion_cache_arr_χ[:, end .- I] .= 1
            end
        elseif InitType == "Rand"
            Random.seed!(seed)
            N = length(motion_cache_arr_χ)
            perm = randperm(N)
            M = round(Int, N * vol)
            motion_cache_arr_χ[ perm[1:M] ] .= 1
        else
            error("InitType not defined!") |> throw
        end
    end

    motion_cache_arr_χ₂ = similar(motion_cache_arr_χ)
    motion_cache_arr_Gτχ = similar(motion_cache_arr_χ)
    motion_cache_arr_Gτχ₂ = similar(motion_cache_arr_χ)
    return motion_cache_arr_χ, motion_cache_arr_χ₂, motion_cache_arr_Gτχ, motion_cache_arr_Gτχ₂
end

"""
return 
    test_spaces: vector of test spaces;                         \\
    trial_spaces: vector of trial spaces;                       \\
    assemblers: vector of assemblers;                           \\
    cache_As: vector of caches for stiffness matrix;            \\
    cache_bs: vector of caches for R.H.S. vector.               \\
    cache_fe_funcs: vector of caches for FEFunction.            \\
    cache_ad_fe_funcs: vector of caches for adjoint FEFunction. 
"""
function initspaces(model, dx, Td, ud, diri_tags::Vector)
    @info "------------- space setting -------------"
    ref_T = ReferenceFE(lagrangian, Float64, 1)
    ref_V = ReferenceFE(lagrangian, VectorValue{2, Float64}, 1)
    ref_P = ReferenceFE(lagrangian, Float64, 1)

    @info "constructing trial and test spaces of heat equation..."
    T_test = TestFESpace(model, ref_T; conformity= :H1, dirichlet_tags= diri_tags[1])
    T_trial = TrialFESpace(T_test, Td)

    @info "constructing trial and test spaces of Stoke equation..."
    V_test = TestFESpace(model, ref_V; conformity= :H1, dirichlet_tags= diri_tags[2])
    V_trial = TrialFESpace(V_test, ud)

    P_test = TestFESpace(model, ref_P; conformity= :H1, constraint= :zeromean)
    P_trial = TrialFESpace(P_test)

    X = MultiFieldFESpace([V_trial, P_trial])
    Y = MultiFieldFESpace([V_test, P_test])

    @info "preparing stiffness matrix cache..."
    T_assem = SparseMatrixAssembler(T_trial, T_test)
    V_assem = SparseMatrixAssembler(X, Y)

    T_cell_dof_ids = get_cell_dof_ids(T_trial);
    X_cell_dof_ids = get_cell_dof_ids(X);

    cache_T_b = allocate_vector(T_assem, (nothing, [T_cell_dof_ids]))
    cache_T_A = allocate_matrix(T_assem, ([[1.]], [T_cell_dof_ids], [T_cell_dof_ids]))

    du, dp = get_trial_fe_basis(X)
    dv, dq = get_fe_basis(Y)
    inte = ∫(du⋅dv + ∇⋅du * dq + ∇⋅dv * dp)dx + ∫(∇(dp)⋅ ∇(dq))dx
    iwq = inte[dx.quad.trian];
    cache_V_b = allocate_vector(V_assem, (nothing, [X_cell_dof_ids]))
    cache_V_A = allocate_matrix(V_assem, ([iwq], [X_cell_dof_ids], [X_cell_dof_ids]))

    @info "preparing cache FEFunction for Th, Thˢ, uh, uhˢ..."
    T_n = get_free_dof_ids(T_trial) |> length
    cache_Th = FEFunction(T_trial, zeros(Float64, T_n))
    cache_Thˢ = FEFunction(T_test, zeros(Float64, T_n))
    V_n = get_free_dof_ids(X) |> length
    cache_X = FEFunction(X, zeros(Float64, V_n))
    cache_Xˢ = FEFunction(Y, zeros(Float64, V_n))
    cache_uh, _ = cache_X 
    cache_uhˢ, _ = cache_Xˢ

    return  (T_test, Y), # test_spaces
            (T_trial, X), # trial_spaces
            (T_assem, V_assem), # assemblers
            (cache_T_A, cache_V_A), # cache_As
            (cache_T_b, cache_V_b), # cache_bs
            (cache_Th, cache_uh), # cache_fe_funcs
            (cache_Thˢ, cache_uhˢ) # cache_ad_fe_funcs
end

"""
lapack solver.
"""
@inline function solver(x, A, b)
    copy!(x, b)
    fa = lu(A)
    ldiv!(fa, x)
    return nothing
end

"""
update 
    motion_cache_arrs:      \\
    - motion_cache_arr_χ    \\
    - motion_cache_arr_χ₂   \\
    - motion_cache_arr_Gτχ  \\
    - motion_cache_arr_Gτχ₂ \\
    motion_cache_fe_funcs:  \\
    - motion_cache_fe_χ     \\
    - motion_cache_fe_Gτχ   \\
    - motion_cache_fe_α     \\
    - motion_cache_fe_κ     \\
and return 
    fe_χ: finite function χ on the whole domain;        \\
    fe_Gτχ: finite function Gτχ on the whole domain;    \\
    fe_α: finite function α on the whole domain;        \\
    fe_κ: finite function κ on the whole domain; 
    
"""
function getcoeff!(
    motion_cache_arrs,
    motion_cache_fe_funcs, 
    fixed_fe_χ,
    motion,
    params,
    perm)

    motion_cache_fe_χ, motion_cache_fe_Gτχ, motion_cache_fe_α, motion_cache_fe_κ = motion_cache_fe_funcs
    motion_cache_arr_χ, motion_cache_arr_χ₂, motion_cache_arr_Gτχ, motion_cache_arr_Gτχ₂ = motion_cache_arrs
    α₋, α⁻, kf, ks = params

    @tturbo @. motion_cache_arr_χ₂ = 1 - motion_cache_arr_χ
    motion(motion_cache_arr_Gτχ, motion_cache_arr_χ)
    motion(motion_cache_arr_Gτχ₂, motion_cache_arr_χ₂)

    for i in eachindex(perm) 
        p_i = perm[i]
        motion_cache_fe_α.free_values[p_i] = α₋ * motion_cache_arr_Gτχ[i] + α⁻ * motion_cache_arr_Gτχ₂[i]
        motion_cache_fe_κ.free_values[p_i] = kf * motion_cache_arr_Gτχ[i] + ks * motion_cache_arr_Gτχ₂[i]
        motion_cache_fe_χ.free_values[p_i] = motion_cache_arr_χ[i]
        motion_cache_fe_Gτχ.free_values[p_i] = motion_cache_arr_Gτχ[i]
    end

    fe_χ = fixed_fe_χ + motion_cache_fe_χ
    fe_Gτχ = fixed_fe_χ + motion_cache_fe_Gτχ
    fe_α = fixed_fe_χ * α₋ + motion_cache_fe_α
    fe_κ = fixed_fe_χ * kf + motion_cache_fe_κ

    return fe_χ, fe_Gτχ, fe_α, fe_κ
end

"""
solve pde and update result `cache_fe_funcs` according to ...
"""
function pde_solve!(
    cache_fe_funcs, 
    test_spaces, 
    trial_spaces, 
    cache_As, 
    cache_bs, 
    assemblers,
    params,
    coeffs,
    dx,
    dΓs)
    
    T_assem, V_assem = assemblers
    T_test, Y = test_spaces
    T_trial, X = trial_spaces
    cache_T_A, cache_V_A = cache_As 
    cache_T_b, cache_V_b = cache_bs
    Th, uh = cache_fe_funcs
    χ, Gτχ, α, κ = coeffs
    dΓin = dΓs[2]

    g, β₁, β₂, β₃, N, Re, δt, γ, Ts, τ = params
    h = 1 / N; δt *= h^2; δu = h^2; μ = 1/Re
    
    a_V((u, p), (v, q)) = ∫(∇(u)⊙∇(v)*μ + u⋅v*α - (∇⋅v)*p + q*(∇⋅u))dx + ∫(∇(p)⋅∇(q)*δu)dx
    l_V((v, q)) = ∫( g ⋅ v)dΓin
    assemble_matrix!(a_V, cache_V_A, V_assem, X, Y)
    assemble_vector!(l_V, cache_V_b, V_assem, Y)
    solver(uh.free_values.parent, cache_V_A, cache_V_b)

    a_T(T, v) = ∫(∇(T) ⋅ ∇(v) * κ + uh⋅∇(T)*v*Re + γ*κ*T*v)dx + ∫((uh⋅∇(T)*Re + γ*κ*T)*(Re*uh⋅∇(v)*δt))dx
    l_T(v) = ∫(γ*κ*Ts*v)*dx + ∫(γ*κ*Ts*Re*uh⋅∇(v)*δt)dx
    assemble_matrix!(a_T, cache_T_A, T_assem, T_trial, T_test)
    assemble_vector!(l_T, cache_T_b, T_assem, T_test)
    solver(Th.free_values, cache_T_A, cache_T_b)
    
    Ju = β₁/2*(μ* ∫(∇(uh)⊙∇(uh))dx + ∫(α*uh⋅uh)dx) |> sum
    Jγ = β₂ * sqrt(π/τ) * ∫(χ * (1 - Gτχ))dx |> sum
    Jt = β₃* ∫((Th - Ts)*κ*γ)dx |> sum
    return Ju, Jγ, Jt
end

"""
solve adjoint pde and update result `cache_fe_funcs` according to ...
"""
function adjoint_pde_solve!(
    cache_ad_fe_funcs,
    cache_fe_funcs,
    test_spaces,
    trial_spaces,
    cache_As,
    cache_bs,
    assemblers,
    params,
    coeffs,
    dx)

    Thˢ, uhˢ = cache_ad_fe_funcs
    T_assem, V_assem = assemblers
    T_test, Y = test_spaces
    T_trial, X = trial_spaces
    cache_T_A, cache_V_A = cache_As 
    cache_T_b, cache_V_b = cache_bs
    Th, uh = cache_fe_funcs
    _, _, α, κ = coeffs
    N, Re, δt, γ = params

    h = 1 / N; δt *= h^2; δu = h^2; μ = 1/Re

    a_Tˢ(Tˢ, v) = ∫(∇(Tˢ) ⋅ ∇(v) * κ + uh⋅∇(v)*Tˢ*Re + γ*κ*Tˢ*v)dx + ∫((uh⋅∇(Tˢ)*Re - γ*κ*Tˢ)*(Re*uh⋅∇(v))*δt)dx 
    l_Tˢ(v) = ∫(- κ *γ * v)dx + ∫(κ *γ * (Re*uh⋅∇(v))*δt)dx
    assemble_matrix!(a_Tˢ, cache_T_A, T_assem, T_trial, T_test)
    assemble_vector!(l_Tˢ, cache_T_b, T_assem, T_test)
    solver(Thˢ.free_values, cache_T_A, cache_T_b)

    a_Vˢ((uˢ, pˢ), (v, q)) = ∫(μ*∇(uˢ)⊙∇(v) + uˢ⋅v*α + (∇⋅v)*pˢ - q*(∇⋅uˢ))dx + ∫(∇(pˢ)⋅∇(q)*δu)dx
    l_Vˢ((v, q)) = ∫(-(∇(Th))⋅v*Re*Thˢ)dx + ∫(-(∇(Th))⋅ ∇(q) *Re*Thˢ * δu)dx
    assemble_matrix!(a_Vˢ, cache_V_A, V_assem, X, Y)
    assemble_vector!(l_Vˢ, cache_V_b, V_assem, Y)   
    solver(uhˢ.free_values.parent, cache_V_A, cache_V_b)

    return nothing
end

"""
implementation of version when Triangulation on 
    GridPortion: a part of the whole domain; \\
    UnstructuredGrid: the whold domain.     \\
"""
@generated function _compute_node_value!(cache::Matrix{T}, f, Ω::BodyFittedTriangulation{Dc, 2, Tm, Tg}) where {T, Dc, Tm, Tg}
    if Tg <: GridPortion
        get_p2c = quote
            node_to_parent_node = Ω.grid.node_to_parent_node
            parent_p2c = get_faces(top, 0, Dc)
            p2c = parent_p2c[node_to_parent_node] 
        end
        node_i = :(node_to_parent_node[i])
    elseif Tg <: UnstructuredGrid
        get_p2c = quote
            p2c = get_faces(top, 0, Dc)
        end
        node_i = :(i)
    else
        error("Triangulation type not supported!")
    end
    
    quote
        model = get_background_model(Ω)
        top = get_grid_topology(model)

        c_p = get_cell_points(Triangulation(model))
        c_val = f(c_p)
        
        $get_p2c # p2c
        c2p = get_faces(top, Dc, 0)
        # traveling vertices
        @inbounds for i = eachindex(p2c)
            cache[i] = zero(T)
            c_ini = p2c.ptrs[i]
            l = p2c.ptrs[i + 1] - c_ini
            c_ini -= one(c_ini)

            # c_ini + j is the jth cell around the ith node
            for j = Base.oneto(l)
                # cell index
                c_i = p2c.data[c_ini + j]

                # p_init is the index of first node of the cell `c_i`
                p_ini = c2p.ptrs[c_i]
                _l = c2p.ptrs[c_i + 1] - p_ini

                p_ini -= one(p_ini)
                idx = 1
                while c2p.data[p_ini + idx] != $node_i
                    idx += 1
                end

                cache[i] += c_val[c_i][idx]
            end
            cache[i] /= l
        end
        nothing
    end
end

"""
compute Φ, note that motion does not have linearity !!!
it could be optimized using `perm`.
"""
function Phi!(
    motion_cache_Φs,
    params,
    cache_fe_funcs,
    cache_ad_fe_funcs,
    motion_space,
    motion,
    motion_cache_arr_Gτχ,
    motion_cache_arr_Gτχ₂)

    cache_Φ, cache_rev_Φ, cache_node_val = motion_cache_Φs
    Th, uh = cache_fe_funcs
    Thˢ, uhˢ = cache_ad_fe_funcs
    β₁, β₂, β₃, α⁻, α₋, Ts, kf, ks, γ = params
    
    τ = motion.τ[]
    trian = get_triangulation(motion_space)

    ## use cache of `cache_rev_Φ` to store result of `motion`
    out = cache_rev_Φ

    # ---------------------- compute Φ₁ - Φ₂ ----------------------
    ## β₁/2 * (α₋ - α⁻) * Gτ(uh⋅uh)
    _compute_node_value!(cache_node_val, uh⋅uh, trian)
    motion(out, cache_node_val)
    @turbo @. cache_Φ = β₁/2 * (α₋ - α⁻) * out

    ## β₂ * √(π/τ) * Gτ(1 - χ)
    @turbo @. cache_Φ += β₂ * sqrt(π / τ) * motion_cache_arr_Gτχ₂

    ## β₂ * √(π/τ) * Gτχ
    @turbo @. cache_Φ += β₂ * sqrt(π / τ) * motion_cache_arr_Gτχ

    ## β₃ * γ * (ks - kf) * Gτ(Ts - Th)
    _compute_node_value!(cache_node_val, Ts - Th, trian)
    motion(out, cache_node_val)
    @turbo @. cache_Φ += β₃ * γ * (ks - kf) * out

    ## (α₋ - α⁻) * Gτ(uh⋅uhˢ)
    _compute_node_value!(cache_node_val, uh⋅uhˢ, trian)
    motion(out, cache_node_val)
    @turbo @. cache_Φ += (α₋ - α⁻) * out

    ## (kf - ks) * ∇(Th)⋅∇(Thˢ)
    _compute_node_value!(cache_node_val, ∇(Th)⋅∇(Thˢ), trian)
    motion(out, cache_node_val)
    @turbo @. cache_Φ += (kf - ks) * out

    ## γ * (kf - ks) * Gτ((Th - Ts) * Thˢ)
    _compute_node_value!(cache_node_val, (Th - Ts) * Thˢ, trian)
    motion(out, cache_node_val)
    @turbo @. cache_Φ += γ * (kf - ks) * out

    @turbo cache_rev_Φ .= cache_Φ
    reverse!(cache_rev_Φ, dims= 2)
    @turbo @. cache_Φ = (cache_Φ + cache_rev_Φ) / 2
    nothing
end

"""
add dirichlet tag for built-in cartesian mesh.
"""
@generated function add_tag_by_filter!(m::CartesianDiscreteModel{D}, _filter, tag_name) where {D}
    @assert D >= 2 "dimension should be greater than 2 but got $D"
    N = D - 2
    M = D - 1
    quote
        top = get_grid_topology(m)
        lab = get_face_labeling(m)
        coors = get_vertex_coordinates(top)
        
        top_faces = get_faces(top, $M, $N)
        face_to_vertices = get_faces(top, $M, 0)
        # faces_d denotes faces_d_to_{d-1}
        @nextract $N faces d -> get_faces(top, d, d-1)

        # tags_d denotes the d-D faces.
        @nexprs $D d -> tags_{d-1} = get_face_entity(lab, d-1)

        # compute a new tag for dirichlet boundary.
        end_tag = lab.tag_to_entities[end-1][1]
        dirichlet_tag = end_tag + one(end_tag)

        is_updated = false

        @inbounds for k = eachindex(face_to_vertices)
            flag = true
            v_ids = face_to_vertices[k]
            barycenter = zero(eltype(coors))
            for v_id in v_ids
                x = coors[v_id]
                flag *= _filter(x)
                flag || break
                barycenter += x
            end
            if flag
                flag *= _filter(barycenter)
            end
            
            @nexprs 1 _ -> if flag 
                is_updated = true
                cur_face_{$M - 1} =  top_faces[k]
                tags_{$M}[k] = dirichlet_tag
                
                # update tags of faces remained.
                @nloops $N i d -> cur_face_d d->begin # pre-expression
                    tags_d[i_d] = dirichlet_tag
                    cur_face_{d-1} = faces_d[ i_d ]
                end begin # body
                    # update tags of the two vertices of a segment.
                    @nexprs 2 l -> tags_0[ cur_face_0[l] ] = dirichlet_tag
                end
            end
        end
        if is_updated
            add_tag!(lab, tag_name, [dirichlet_tag])
        else
            error("`filter` function is not used, please check the boundary condition.")
        end
        nothing
    end 
end

