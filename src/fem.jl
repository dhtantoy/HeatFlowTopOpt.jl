function initmodel(model::DiscreteModel, boundary_tags::Vector)
    # @info "generating triangulation ..."
    Ω = Triangulation(model)
    Γs = map(boundary_tags) do tags 
        BoundaryTriangulation(model; tags= tags)
    end 

    # @info "generating measure ..."
    dx = Measure(Ω, 4)
    dΓs = map(Γs) do Γ
        Measure(Γ, 4)
    end

    aux_space = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity= :H1)
    
    return aux_space, (Ω, Γs), (dx, dΓs)
end


function initfefuncs(aux_space)
    # @info "constructing fefunction fe_χ, fe_Gτχ fe_α, fe_κ ..."
    cache_fe_χ = FEFunction(aux_space, zeros(Float64, num_free_dofs(aux_space)))
    cache_fe_Gτχ = FEFunction(aux_space, zeros(Float64, num_free_dofs(aux_space)))
    cache_fe_α = FEFunction(aux_space, zeros(Float64, num_free_dofs(aux_space)))
    cache_fe_κ = FEFunction(aux_space, zeros(Float64, num_free_dofs(aux_space)))

    return cache_fe_χ, cache_fe_Gτχ, cache_fe_α, cache_fe_κ
end

function initcachechis(InitType, aux_space; vol= 0.4, seed= 0, file="", key="")
    dim = get_triangulation(aux_space) |> num_point_dims
    np = num_free_dofs(aux_space)
    N_node::Int = np ^ (1//dim)

    # @info "------------- generate χ₀ -------------"
    if InitType == "All"
        cache_arr_χ = ones(Float64, repeat([N_node], dim)...)
    elseif InitType == "File"
        isfile(file) || error("file not found! [$file]") |> throw
        _f = load(file)
        haskey(_f, key) || error("key not found! [$key]") |> throw
        cache_arr_χ = _f[key]
    else
        cache_arr_χ = zeros(Float64, repeat([N_node], dim)...)

        if InitType == "Net"
            n = 20
            p = Iterators.partition(1:(N_node >> 1), N_node ÷ n) |> collect
            for I in p[2:4:end]
                cache_arr_χ[:, I] .= 1
                cache_arr_χ[:, end .- I] .= 1
                cache_arr_χ[I, :] .= 1
                cache_arr_χ[end .- I, :] .= 1
            end
        elseif InitType == "Line"
            n = 20
            p = Iterators.partition(1:(N_node >> 1), N_node ÷ n) |> collect
            for I in p[2:4:end]
                cache_arr_χ[:, I] .= 1
                cache_arr_χ[:, end .- I] .= 1
            end
        elseif InitType == "Rand"
            m₁, m₂ = size(cache_arr_χ)
            c::Int = ceil(m₂ / 2)
            f::Int = floor(m₂ / 2)
            N = c * m₁
            perm = randperm(Random.seed!(seed), N)
            M = round(Int, N * vol)
            cache_arr_χ[ perm[1:M] ] .= 1

            cache_arr_χ[:, c+1:end] = cache_arr_χ[:, f:-1:1]
        else
            error("InitType not defined!") |> throw
        end
    end

    cache_arr_Gτχ = zero(cache_arr_χ)
    return cache_arr_χ, cache_arr_Gτχ
end

"""
return 
- test_spaces: vector of test spaces;                          
- trial_spaces: vector of trial spaces;                        
- assemblers: vector of assemblers;                            
- cache_As: vector of caches for stiffness matrix;             
- cache_bs: vector of caches for R.H.S. vector.                
- cache_fe_funcs: vector of caches for FEFunction.             
- cache_ad_fe_funcs: vector of caches for adjoint FEFunction. 
"""
function initspaces(model, dx, Td, ud)
    # @info "------------- space setting -------------"
    ref_T = ReferenceFE(lagrangian, Float64, 1)
    ref_V = ReferenceFE(lagrangian, VectorValue{2, Float64}, 2)
    ref_P = ReferenceFE(lagrangian, Float64, 1)

    # @info "constructing trial and test spaces of heat equation..."
    T_test = TestFESpace(model, ref_T; conformity= :H1, dirichlet_tags= [7])
    T_trial = TrialFESpace(T_test, Td)

    # @info "constructing trial and test spaces of Stoke equation..."
    V_test = TestFESpace(model, ref_V; conformity= :H1, dirichlet_tags= [5, 6])
    V_trial = TrialFESpace(V_test, ud)

    P_test = TestFESpace(model, ref_P; conformity= :H1, constraint= :zeromean)
    P_trial = TrialFESpace(P_test)

    X = MultiFieldFESpace([V_trial, P_trial])
    Y = MultiFieldFESpace([V_test, P_test])

    # @info "preparing stiffness matrix cache..."
    T_assem = SparseMatrixAssembler(T_trial, T_test)
    V_assem = SparseMatrixAssembler(X, Y)

    T_cell_dof_ids = get_cell_dof_ids(T_trial);
    X_cell_dof_ids = get_cell_dof_ids(X);

    du = get_trial_fe_basis(T_trial)
    dv = get_fe_basis(T_test)
    iwq_T_A = integrate(du * dv, dx.quad)
    rs_T_A = ([iwq_T_A], [T_cell_dof_ids], [T_cell_dof_ids])
    cache_T_b = allocate_vector(T_assem, (nothing, [T_cell_dof_ids]))
    cache_T_A = allocate_matrix(T_assem, rs_T_A)
    assemble_matrix!(cache_T_A, T_assem, rs_T_A)
    cache_T_LU = lu(cache_T_A)

    du, dp = get_trial_fe_basis(X)
    dv, dq = get_fe_basis(Y)
    iwq_V_A = integrate(du⋅dv + ∇⋅du * dq + ∇⋅dv * dp, dx.quad)
    # iwq_V_A = integrate(du⋅dv + ∇⋅du * dq + ∇⋅dv * dp + ∇(dp)⋅ ∇(dq), dx.quad)
    rs_V_A = ([iwq_V_A], [X_cell_dof_ids], [X_cell_dof_ids])
    cache_V_b = allocate_vector(V_assem, (nothing, [X_cell_dof_ids]))
    cache_V_A = allocate_matrix(V_assem, rs_V_A)
    assemble_matrix!(cache_V_A, V_assem, rs_V_A)
    cache_V_LU = lu(cache_V_A)

    # @info "preparing cache FEFunction for Th, Thˢ, uh, uhˢ..."
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
            (cache_T_LU, cache_V_LU), # cache_LUs
            (cache_T_b, cache_V_b), # cache_bs
            (cache_Th, cache_uh), # cache_fe_funcs
            (cache_Thˢ, cache_uhˢ) # cache_ad_fe_funcs
end

"""
lapack solver.
"""
@inline function solver(x, A, fa, b)
    copy!(x, b)
    lu!(fa, A)
    ldiv!(fa, x)
    return nothing
end


function update_motion_funcs!(
    motion_funcs,
    cache_arrs,
    motion,
    params)

    fe_χ, fe_Gτχ, fe_α, fe_κ = motion_funcs
    cache_arr_χ, cache_arr_χ₂, cache_arr_Gτχ, cache_arr_Gτχ₂ = cache_arrs
    α⁻, kf, ks = params

    @turbo @. cache_arr_χ₂ = 1 - cache_arr_χ
    motion(cache_arr_Gτχ, cache_arr_χ)
    motion(cache_arr_Gτχ₂, cache_arr_χ₂)
    

    @turbo for i = eachindex(fe_α.free_values)
        fe_α.free_values[i] = α⁻ * cache_arr_Gτχ₂[i]
    end

    @turbo for i = eachindex(fe_κ.free_values)
        fe_κ.free_values[i] = kf * cache_arr_Gτχ[i] + ks * cache_arr_Gτχ₂[i]
    end

    # @turbo @. fe_α.free_values = α⁻ * (1 - cache_arr_Gτχ)
    # @turbo @. fe_κ.free_values = (kf - ks) * cache_arr_Gτχ + ks
    copy!(fe_χ.free_values, vec(cache_arr_χ))
    copy!(fe_Gτχ.free_values, vec(cache_arr_Gτχ))

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
    cache_lus,
    cache_bs, 
    assemblers,
    params,
    motion_funcs,
    dx,
    dΓs)
    
    T_assem, V_assem = assemblers
    T_test, Y = test_spaces
    T_trial, X = trial_spaces
    cache_T_A, cache_V_A = cache_As 
    cache_T_LU, cache_V_LU = cache_lus
    cache_T_b, cache_V_b = cache_bs
    Th, uh = cache_fe_funcs
    χ, Gτχ, α, κ = motion_funcs
    dΓin = dΓs[2]

    g, β₁, β₂, β₃, N, Re, δt, δu, γ, Ts, τ = params
    h = 1 / N; δt *= h^2; μ = 1/Re; δu *= h^2;
    
    a_V((u, p), (v, q)) = ∫(∇(u)⊙∇(v)*μ + u⋅v*α - (∇⋅v)*p + q*(∇⋅u))dx # + ∫(∇(p)⋅∇(q)*δu)dx
    l_V((v, q)) = ∫( g ⋅ v)dΓin
    assemble_matrix!(a_V, cache_V_A, V_assem, X, Y)
    assemble_vector!(l_V, cache_V_b, V_assem, Y)
    solver(uh.free_values.parent, cache_V_A, cache_V_LU, cache_V_b)

    a_T(T, v) = ∫(∇(T) ⋅ ∇(v) * κ + uh⋅∇(T)*v*Re + γ*κ*T*v)dx + ∫((uh⋅∇(T)*Re + γ*κ*T)*(Re*uh⋅∇(v)*δt))dx
    l_T(v) = ∫(γ*κ*Ts*v)*dx + ∫(γ*κ*Ts*Re*uh⋅∇(v)*δt)dx
    assemble_matrix!(a_T, cache_T_A, T_assem, T_trial, T_test)
    assemble_vector!(l_T, cache_T_b, T_assem, T_test)
    solver(Th.free_values, cache_T_A, cache_T_LU, cache_T_b)
    
    Ju = β₁/2*(μ* ∫(∇(uh)⊙∇(uh))dx + ∫(α*uh⋅uh)dx) |> sum
    Jγ = β₂ * sqrt(π/τ) * ∫(χ * (1 - Gτχ))dx |> sum

    isnan(Jγ) && (Jγ = 0) 
    isinf(Jγ) && (throw(DivideError()))

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
    cache_lus,
    cache_bs,
    assemblers,
    params,
    motion_funcs,
    dx)

    Thˢ, uhˢ = cache_ad_fe_funcs
    T_assem, V_assem = assemblers
    T_test, Y = test_spaces
    T_trial, X = trial_spaces
    cache_T_A, cache_V_A = cache_As 
    cache_T_LU, cache_V_LU = cache_lus
    cache_T_b, cache_V_b = cache_bs
    Th, uh = cache_fe_funcs
    _, _, α, κ = motion_funcs
    N, Re, δt, δu, γ, β₃ = params

    h = 1 / N; δt *= h^2; μ = 1/Re; δu *= h^2; 

    a_Tˢ(Tˢ, v) = ∫(∇(Tˢ) ⋅ ∇(v) * κ + uh⋅∇(v)*Tˢ*Re + γ*κ*Tˢ*v)dx + ∫((uh⋅∇(Tˢ)*Re - γ*κ*Tˢ)*(Re*uh⋅∇(v))*δt)dx 
    l_Tˢ(v) = ∫(- β₃ * κ *γ * v)dx + ∫(β₃ * κ *γ * (Re*uh⋅∇(v))*δt)dx
    assemble_matrix!(a_Tˢ, cache_T_A, T_assem, T_trial, T_test)
    assemble_vector!(l_Tˢ, cache_T_b, T_assem, T_test)
    solver(Thˢ.free_values, cache_T_A, cache_T_LU, cache_T_b)

    a_Vˢ((uˢ, pˢ), (v, q)) = ∫(μ*∇(uˢ)⊙∇(v) + uˢ⋅v*α + (∇⋅v)*pˢ - q*(∇⋅uˢ))dx #+ ∫(∇(pˢ)⋅∇(q)*δu)dx
    l_Vˢ((v, q)) = ∫(-(∇(Th))⋅v*Re*Thˢ)dx #+ ∫(-(∇(Th))⋅ ∇(q) *Re*Thˢ * δu)dx
    assemble_matrix!(a_Vˢ, cache_V_A, V_assem, X, Y)
    assemble_vector!(l_Vˢ, cache_V_b, V_assem, Y)   
    solver(uhˢ.free_values.parent, cache_V_A, cache_V_LU, cache_V_b)

    return nothing
end


"""
implementation of version when Triangulation on 
    GridPortion: a part of the whole domain;  
    UnstructuredGrid: the whold domain.   
it(cell-wise) evaluates faster than that(parallelly point-wise).
"""

function _compute_node_value!(out, op, trian)
    T = eltype(out)
    Dc = num_cell_dims(trian)
    fill!(out, zero(T))

    model = get_background_model(trian)
    top = get_grid_topology(model)
    c_p = get_cell_points(Triangulation(model))
    c_val = op(c_p)
    
    c2p = get_faces(top, Dc, 0)
    counts = zeros(eltype(c2p.ptrs), size(out)...)

    @inbounds for c_i = eachindex(c2p)
        cache = c_val[c_i]
        p_ini = c2p.ptrs[c_i]
        l = c2p.ptrs[c_i + 1] - p_ini   
        p_ini -= one(p_ini)
        for j = Base.oneto(l)
            p_i = c2p.data[p_ini + j]
            out[p_i] += cache[j]
            counts[p_i] += one(p_ini)
        end
    end
    out ./= counts
    return nothing
end


"""
compute Φ, note that motion does not have linearity !!!
it could be optimized using `perm`.
"""
function Phi!(
    cache_Φs,
    params,
    cache_fe_funcs,
    cache_ad_fe_funcs,
    aux_space,
    motion,
    cache_arr_Gτχ,
    cache_arr_Gτχ₂)

    cache_Φ, cache_rev_Φ, cache_node_val = cache_Φs
    Th, uh = cache_fe_funcs
    Thˢ, uhˢ = cache_ad_fe_funcs
    β₁, β₂, β₃, α⁻, Ts, kf, ks, γ = params
    
    τ = motion.τ[]
    trian = get_triangulation(aux_space)

    # ---------------------- compute Φ₁ - Φ₂ ----------------------
    _M = -α⁻*(β₁/2*uh⋅uh + 2*uh⋅uhˢ) + (kf - ks)*(∇(Th)⋅∇(Thˢ)) + γ*(ks - kf)*((Ts - Th)*(Thˢ + β₃))
    _compute_node_value!(cache_node_val, _M, trian)
    motion(cache_Φ, cache_node_val)

    r = β₂ * sqrt(π / τ)
    isnan(r) && (r = 0)
    isinf(r) && (throw(DivideError()))

    @turbo @. cache_Φ += r * (cache_arr_Gτχ₂ - cache_arr_Gτχ)
    
    copy!(cache_rev_Φ, cache_Φ)
    reverse!(cache_rev_Φ, dims= 2)
    @turbo @. cache_Φ = (cache_Φ + cache_rev_Φ) / 2
    nothing
end

function Phi_debug(
    motion_cache_Φs,
    params,
    cache_fe_funcs,
    cache_ad_fe_funcs,
    aux_space,
    motion,
    cache_arr_Gτχ,
    cache_arr_Gτχ₂)

    _, cache_rev_Φ, cache_node_val = motion_cache_Φs
    Th, uh = cache_fe_funcs
    Thˢ, uhˢ = cache_ad_fe_funcs
    β₁, β₂, β₃, α⁻, Ts, kf, ks, γ = params
    
    τ = motion.τ[]
    trian = get_triangulation(aux_space)

    ## use cache of `cache_rev_Φ` to store result of `motion`
    out = cache_rev_Φ

    # ---------------------- compute Φ₁ - Φ₂ ----------------------
    ## β₁/2 * (- α⁻) * Gτ(uh⋅uh)
    _compute_node_value!(cache_node_val, uh⋅uh, trian)
    motion(out, cache_node_val)
    ret1 = β₁/2 * (- α⁻) * out


    ## β₂ * √(π/τ) * Gτ(1 - χ)
    ret2 = @. β₂ * sqrt(π / τ) * cache_arr_Gτχ₂

    ## - β₂ * √(π/τ) * Gτχ
    ret3 = β₂ * sqrt(π / τ) * cache_arr_Gτχ

    ## β₃ * γ * (ks - kf) * Gτ(Ts - Th)
    _compute_node_value!(cache_node_val, Ts - Th, trian)
    motion(out, cache_node_val)
    ret4 = β₃ * γ * (ks - kf) * out

    ## (- α⁻) * Gτ(uh⋅uhˢ)
    _compute_node_value!(cache_node_val, uh⋅uhˢ, trian)
    motion(out, cache_node_val)
    ret5 = (- α⁻) * out

    ## (kf - ks) * Gτ(∇T⋅∇Tˢ)
    _compute_node_value!(cache_node_val, ∇(Th)⋅∇(Thˢ), trian)
    motion(out, cache_node_val)
    ret6 = (kf - ks) * out

    ## γ * (ks - kf) * Gτ((Ts - Th) * Thˢ)
    _compute_node_value!(cache_node_val, (Ts - Th) * Thˢ, trian)
    motion(out, cache_node_val)
    ret7 = γ * (ks - kf) * out

    return ret1, ret2, ret3, ret4, ret5, ret6, ret7
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

