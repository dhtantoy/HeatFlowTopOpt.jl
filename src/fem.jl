
"""
return 
    model: grid and topology of grid;
    motion_space: fe space based on computational domain;
    fixed_space: fe space based on the rest domain;
    Ω: Triangulation of model;
    Γs: BoundaryTriangulation of model;
    dx: Measure on Ω;
    dΓs: Measures on Γs;
    perm: permutation map generated from `perm_map_to_cartesian(aux_Ω)`.
"""
function initgmshmodel(N, L, motion_domain_tags::Vector, fixed_domain_tags::Vector, boundary_tags::Vector)
    @info "------------- model setting -------------"
    
    prefix = "models"
    mkpath(prefix)

    n, r = divrem(N, 3)
    iszero(r) || error("N should be divided by 3!")

    file_path = joinpath(prefix, "N_$(N)_L_$L.msh")
    isfile(file_path) || begin
        @warn "file not found, creating a new one as $file_path ..."
        createGrid(Val(2), n , L, file_path)
    end

    @info "import gmsh file $file_path ..."
    model = DiscreteModelFromFile(file_path)

    @info "generating triangulation ..."
    Ω = Triangulation(model)
    motion_Ω = Triangulation(model, tags= motion_domain_tags)
    fixed_Ω = Triangulation(model, tags= fixed_domain_tags)
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

    @info "constructing fefunction fe_χ, fe_Gτχ ..."
    motion_cache_fe_χ = FEFunction(motion_space, zeros(Float64, num_free_dofs(motion_space)))
    motion_cache_fe_Gτχ = FEFunction(motion_space, zeros(Float64, num_free_dofs(motion_space)))

    return model, motion_space, fixed_space, Ω, Γs, dx, dΓs, (perm, motion_cache_fe_χ, motion_cache_fe_Gτχ)
end

"""
generate χ₀ with size `N_node` × `N_node` × ... × `N_node` and dimension `dim`.
`InitType` could be `:Net` or `:Line`.
"""
function initchi(InitType, N_node, dim, n= 20)
    @info "------------- generate χ₀ -------------"
    if InitType == "All"
        χ = ones(Float64, repeat([N_node], dim)...)
    else
        χ = zeros(Float64, repeat([N_node], dim)...)
        p = Iterators.partition(1:(N_node >> 1), N_node ÷ n) |> collect

        if InitType == "Net"
            for I in p[2:4:end]
                χ[:, I] .= 1
                χ[:, end .- I] .= 1
                χ[I, :] .= 1
                χ[end .- I, :] .= 1
            end
        elseif InitType == "Line"
            for I in p[2:4:end]
                χ[:, I] .= 1
                χ[:, end .- I] .= 1
            end
        else 
            error("InitType not defined!") |> throw
        end
    end

    return χ
end

"""
it should be overwrite with `import HeatFlowTopOpt: initspaces`, and 
return 
    test_spaces: vector of test spaces;
    trial_spaces: vector of trial spaces;
    assemblers: vector of assemblers;
    cache_As: vector of caches for stiffness matrix;
    cache_bs: vector of caches for R.H.S. vector.
    cache_fe_funcs: vector of caches for FEFunction.
    cache_ad_fe_funcs: vector of caches for adjoint FEFunction.

default for heat flow problem. 
"""
function initspaces(model, dx, Td, ud)
    @info "------------- space setting -------------"
    ref_T = ReferenceFE(lagrangian, Float64, 1)
    ref_V = ReferenceFE(lagrangian, VectorValue{2, Float64}, 1)
    ref_P = ReferenceFE(lagrangian, Float64, 1)

    @info "constructing trial and test spaces of heat equation..."
    T_test = TestFESpace(model, ref_T; conformity= :H1, dirichlet_tags= [2])
    T_trial = TrialFESpace(T_test, Td)

    @info "constructing trial and test spaces of Stoke equation..."
    V_test = TestFESpace(model, ref_V; conformity= :H1, dirichlet_tags= [1])
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
    iwq = (∫(du⋅dv + ∇⋅du * dq + ∇⋅dv * dp + ∇(dp)⋅ ∇(dq))dx)[dx.quad.trian];
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


@inline function solver(x, A, b)
    copy!(x, b)
    fa = lu(A)
    ldiv!(fa, x)
    return nothing
end

"""
return (fe_χ, fe_κ, fe_α, fe_Gτχ)
"""
function _coeff_cache(cache_for_fe, fixed_fe_χ, motion_cache_χ, motion, params)
    α₋, α⁻, kf, ks = params
    perm, motion_cache_fe_χ, motion_cache_fe_Gτχ = cache_for_fe

    motion(motion_cache_χ)
    motion_cache_fe_χ.free_values[perm] = motion_cache_χ
    motion_cache_fe_Gτχ.free_values[perm] = motion.out
    fe_Gτχ = motion_cache_fe_Gτχ + fixed_fe_χ
    fe_χ = motion_cache_fe_χ + fixed_fe_χ

    fe_κ = ks + (kf - ks) * fe_Gτχ
    fe_α = α⁻ + (α₋ - α⁻) * fe_Gτχ

    return (fe_χ, fe_κ, fe_α, fe_Gτχ)
end

function pde_solve!(
    cache_fe_funcs, 
    test_spaces, 
    trial_spaces, 
    cache_As, 
    cache_bs, 
    assemblers,
    params,
    cache_coeff,
    motion,
    dx,
    dΓs)
    
    T_assem, V_assem = assemblers
    T_test, Y = test_spaces
    T_trial, X = trial_spaces
    cache_T_A, cache_V_A = cache_As 
    cache_T_b, cache_V_b = cache_bs
    Th, uh = cache_fe_funcs
    χ, κ, α, Gτχ = cache_coeff
    dΓin = dΓs[2]

    g, β₁, β₂, β₃, N, Re, δt, γ, Ts = params
    τ = motion.τ[]; h = 1 / N; δt *= h^2; δu = h^2; μ = 1/Re
    
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


function adjoint_pde_solve!(
    cache_ad_fe_funcs,
    cache_fe_funcs,
    test_spaces,
    trial_spaces,
    cache_As,
    cache_bs,
    assemblers,
    params,
    cache_coeff,
    dx)

    Thˢ, uhˢ = cache_ad_fe_funcs
    T_assem, V_assem = assemblers
    T_test, Y = test_spaces
    T_trial, X = trial_spaces
    cache_T_A, cache_V_A = cache_As 
    cache_T_b, cache_V_b = cache_bs
    Th, uh = cache_fe_funcs
    _, κ, α, _ = cache_coeff
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
parallel version. but need to debug
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


# @generated function _compute_node_value!(cache::Matrix{T}, f, Ω::BodyFittedTriangulation{Dc, 2, Tm, Tg}) where {T, Dc, Tm, Tg}
#     quote
#         ns = zeros(Int, size(cache))
#         c_val = f(c_p)

#         for i = eachindex(c2p)
#             c_ini = c2p.ptrs[i]
#             l = c2p.ptrs[i + 1] - c_ini
#             c_ini -= one(c_ini)
#             for j = Base.oneto(l)
#                 c_i = c2p.data[c_ini + j]
#                 ns[p_i] += 1
#                 cache[p_i] += c_val[]
#             end
#         end
#     end

# end

function Phi!(
    motion_cache_Φs,
    params,
    cache_fe_funcs,
    cache_ad_fe_funcs,
    motion_space,
    motion)

    cache_Φ, cache_rev_Φ, cache_node_val = motion_cache_Φs
    Th, uh = cache_fe_funcs
    Thˢ, uhˢ = cache_ad_fe_funcs
    β₁, β₂, β₃, α⁻, α₋, Ts, kf, ks, γ = params
    τ = motion.τ[]

    Ω = get_triangulation(motion_space)

    copy!(cache_Φ, motion.out)
    c = β₂ * sqrt(π / τ)
    @turbo @. cache_Φ = c * (1 - 2 * cache_Φ)

    # first term should be negative
    # withou h^2
    f = -(α⁻ - α₋)*(β₁/2 * uh⋅uh + uh⋅uhˢ) + (kf - ks)*∇(Th)⋅∇(Thˢ) + γ*(Th - Ts) * (kf - ks) *(β₃ + Thˢ)
    
    _compute_node_value!(cache_node_val, f, Ω)
    motion(cache_node_val)
    @turbo cache_Φ .+= motion.out
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

