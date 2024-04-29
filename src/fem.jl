"""
return 
    m, lab, cache_χ, aux_space
    (Ω, Γin, Γout, Γwall),
    (dx, dΓin, dΓout, dΓwall)
"""
function initmodel(::Val{InitType}, N, dim, L) where {InitType}

    @info "------------- grid setting -------------"

    @info "generating grid, qudrature and χ₀..."
    m = CartesianDiscreteModel(repeat([0, L], dim), repeat([N], dim))|> simplexify
    lab = get_face_labeling(m)
    add_tag!(lab, "in", [7])
    add_tag!(lab, "out", [8])
    add_tag!(lab, "wall", [1, 2, 3, 4, 5, 6])

    Ω = Triangulation(m)
    Γin = BoundaryTriangulation(m; tags= "in")
    Γout = BoundaryTriangulation(m; tags= "out")
    Γwall = BoundaryTriangulation(m; tags= "wall")

    dx = Measure(Ω, 4)
    dΓin = Measure(Γin, 4)
    dΓout = Measure(Γout, 4)
    dΓwall = Measure(Γwall, 4)

    cache_χ = initchi(Val(InitType), N + 1, dim)

    aux_space = TestFESpace(m, ReferenceFE(lagrangian, Float64, 1); conformity=:H1)

    @info "-------------------------------------------"
    
    m, lab, cache_χ, aux_space,
    (Ω, Γin, Γout, Γwall),
    (dx, dΓin, dΓout, dΓwall)
end

@generated function initchi(::Val{InitType}, N_node, dim, n= 20) where {InitType}
    if InitType == :Net
        op = quote
            χ[I, :] .= 1
            χ[end .- I, :] .= 1
        end
    elseif InitType == :Line 
        op = :(nothing)
    end

    quote
        χ = zeros(Float64, repeat([N_node], dim)...)
        p = Iterators.partition(1:(N_node >> 1), N_node ÷ n) |> collect

        for I in p[2:4:end]
            χ[:, I] .= 1
            χ[:, end .- I] .= 1
            $op
        end

        return χ
    end
end

"""
return 
    (T_test, T_trial, X, Y), 
    (T_assem, V_assem, cache_T_b, cache_T_A, cache_V_b, cache_V_A),
    (cache_Th, cache_Thˢ, cache_uh, cache_uhˢ)
"""
function initspace(m, lab, dx, ud, Td)
    @info "------------- space setting -------------"
    ref_T = ReferenceFE(lagrangian, Float64, 1)
    ref_V = ReferenceFE(lagrangian, VectorValue{2, Float64}, 1)
    ref_P = ReferenceFE(lagrangian, Float64, 1)

    @info "constructing trial and test spaces of heat equation..."
    T_test = TestFESpace(m, ref_T, labels= lab; conformity= :H1, dirichlet_tags= ["in"])
    T_trial = TrialFESpace(T_test, Td)

    @info "constructing trial and test spaces of Stoke equation..."
    V_test = TestFESpace(m, ref_V, labels= lab; conformity= :H1, dirichlet_tags= ["wall"])
    V_trial = TrialFESpace(V_test, ud)

    P_test = TestFESpace(m, ref_P, labels= lab; conformity= :H1, constraint= :zeromean)
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
    iwq = (∫(du⋅dv + ∇⋅du * dq + ∇⋅dv * dp + ∇(dp)⋅ ∇(dq))dx)[dx.quad.trian];
    cache_V_b = allocate_vector(V_assem, (nothing, [X_cell_dof_ids]))
    cache_V_A = allocate_matrix(V_assem, ([iwq], [X_cell_dof_ids], [X_cell_dof_ids]))

    @info "preparing cache FEFunction for Th, Thˢ, uh, uhˢ..."
    T_n = get_free_dof_ids(T_trial) |> length
    cache_Th = FEFunction(T_trial, zeros(Float64, T_n))
    cache_Thˢ = FEFunction(T_trial, zeros(Float64, T_n))
    V_n = get_free_dof_ids(X) |> length
    cache_X = FEFunction(X, zeros(Float64, V_n))
    cache_Xˢ = FEFunction(X, zeros(Float64, V_n))
    cache_uh, _ = cache_X 
    cache_uhˢ, _ = cache_Xˢ
   
    @info "-------------------------------------------"

    (T_test, T_trial, X, Y), 
    (T_assem, V_assem, cache_T_b, cache_T_A, cache_V_b, cache_V_A),
    (cache_Th, cache_Thˢ, cache_uh, cache_uhˢ)
end

@inline function solver(x, A, b)
    copy!(x, b)
    fa = lu(A)
    ldiv!(fa, x)
    return nothing
end

"""
return (χ, κ, α, Gτχ)
"""
function _coeff_cache(cache_χ, conv, aux_space, α⁻)
    α₋ = 0.4175 * 0
    kf = 0.1624
    ks = 40.47
    conv(cache_χ)
    Gτχ_arr = conv.out
    
    Gτχ = FEFunction(aux_space, vec(Gτχ_arr)) 
    χ = FEFunction(aux_space, vec(cache_χ)) 
    κ = ks + (kf - ks) * Gτχ
    α = α⁻ + (α₋ - α⁻) * Gτχ

    return (χ, κ, α, Gτχ)
end

function pde_solve!(cache_fem, cache_fefunc, cache_coeff, params, spaces, dx, dΓin, conv)
    T_assem, V_assem, cache_T_b, cache_T_A, cache_V_b, cache_V_A = cache_fem
    T_test, T_trial, X, Y = spaces
    Th, _, uh, _ = cache_fefunc
    τ = conv.τ[]
    g, β₁, β₂, β₃, N, Re, _, δt = params
    χ, κ, α, Gτχ = cache_coeff

    h = 1 / N
    δt *= h^2
    δu = h^2
    μ = 1/Re
    γ = 1027.6
    Ts = 1.

    a_V((u, p), (v, q)) = ∫(∇(u)⊙∇(v)*μ + u⋅v*α - (∇⋅v)*p + q*(∇⋅u))dx + ∫(∇(p)⋅∇(q)*δu)dx
    l_V((v, q)) = ∫( g ⋅ v)dΓin
    assemble_matrix!(a_V, cache_V_A, V_assem, X, Y)
    assemble_vector!(l_V, cache_V_b, V_assem, Y)
    solver(uh.free_values.vector, cache_V_A, cache_V_b)

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


function adjoint_pde_solve!(cache_fem, cache_fefunc, cache_coeff, params, spaces, dx)
    T_assem, V_assem, cache_T_b, cache_T_A, cache_V_b, cache_V_A = cache_fem
    Th, Thˢ, uh, uhˢ = cache_fefunc
    T_test, T_trial, X, Y = spaces
    _..., N, Re, _, δt = params
    _, κ, α, _ = cache_coeff

    h = 1 / N
    δt *= h^2
    δu = h^2
    μ = 1/Re
    γ = 1027.6

    a_Tˢ(Tˢ, v) = ∫(∇(Tˢ) ⋅ ∇(v) * κ + uh⋅∇(v)*Tˢ*Re + γ*κ*Tˢ*v)dx + ∫((uh⋅∇(Tˢ)*Re - γ*κ*Tˢ)*(Re*uh⋅∇(v))*δt)dx 
    l_Tˢ(v) = ∫(- κ *γ * v)dx + ∫(κ *γ * (Re*uh⋅∇(v))*δt)dx
    assemble_matrix!(a_Tˢ, cache_T_A, T_assem, T_trial, T_test)
    assemble_vector!(l_Tˢ, cache_T_b, T_assem, T_test)
    solver(Thˢ.free_values, cache_T_A, cache_T_b)

    a_Vˢ((uˢ, pˢ), (v, q)) = ∫(μ*∇(uˢ)⊙∇(v) + uˢ⋅v*α + (∇⋅v)*pˢ - q*(∇⋅uˢ))dx + ∫(∇(pˢ)⋅∇(q)*δu)dx
    l_Vˢ((v, q)) = ∫(-(∇(Th))⋅v*Re*Thˢ)dx + ∫(-(∇(Th))⋅ ∇(q) *Re*Thˢ * δu)dx
    assemble_matrix!(a_Vˢ, cache_V_A, V_assem, X, Y)
    assemble_vector!(l_Vˢ, cache_V_b, V_assem, Y)   
    solver(uhˢ.free_values.vector, cache_V_A, cache_V_b)

    return nothing
end


function _compute_node_value!(cache::Matrix{T}, f, Ω::Triangulation{D}) where {T, D}
    c_p = get_cell_points(Ω)
    c_val = f(c_p)
    model = get_background_model(Ω)
    top = get_grid_topology(model)
    p2c = get_faces(top, 0, D)
    c2p = get_faces(top, D, 0)
    # traveling  vertices
    @inbounds for i = eachindex(p2c)
        cache[i] = zero(T)
        c_ini = p2c.ptrs[i]
        l = p2c.ptrs[i + 1] - c_ini
        c_ini -= one(c_ini)
        for j = Base.oneto(l)
            c_i = p2c.data[c_ini + j]
            p_ini = c2p.ptrs[c_i]
            idx = findnext(isequal(i), c2p.data, p_ini) + 1 - p_ini
            cache[i] += c_val[c_i][idx]
        end
        cache[i] /= l
    end
    nothing
end

"""
compute Phi.
(cache_Φ, cache_rev_Φ, cache_node_val) = cache.
(Th, Thˢ, uh, uhˢ) = cache_fefunc.
"""
function Phi!(cache, params, cache_fefunc, Ω, conv)
    cache_Φ, cache_rev_Φ, cache_node_val = cache
    Th, Thˢ, uh, uhˢ = cache_fefunc
    _, β₁, β₂, β₃, N, _, α⁻, _ = params
    τ = conv.τ[]
    α₋ = 0.4175 * 0
    Ts = 1.
    kf = 0.1624
    ks = 40.47
    γ = 1027.6

    copy!(cache_Φ, conv.out)
    c = β₂ * sqrt(π / τ)
    @turbo @. cache_Φ = c * (1 - 2 * cache_Φ)

    # first term should be negative
    # withou h^2
    f = -(α⁻ - α₋)*(β₁/2 * uh⋅uh + uh⋅uhˢ) + (kf - ks)*∇(Th)⋅∇(Thˢ) + γ*(Th - Ts) * (kf - ks) *(β₃ + Thˢ)
    
    _compute_node_value!(cache_node_val, f, Ω)
    conv(cache_node_val)
    @turbo cache_Φ .+= conv.out
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
            for v_id in v_ids
                x = coors[v_id]
                flag *= _filter(x)
                flag || break
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

