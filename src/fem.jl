"""
    init_chi!(cache_arr_χ, InitType; kwargs...)
in-place initialize `cache_arr_χ` according to `InitType`.

# Arguments
- vol::Float64: volume of the initial region. default 0.4.
- seed::Int: seed for random initialization. default 0.
- file::String: file path for initialization.
- key::String: key for initialization.
"""
function init_chi!(cache_arr_χ::Array{T}, InitType; vol= 0.4, seed= 0, file="", key="") where T
    N_node = size(cache_arr_χ, 1)
    if InitType == "All"
        fill!(cache_arr_χ, one(T))
    elseif InitType == "File"
        isfile(file) || error("file not found! [$file]") |> throw
        _f = load(file)
        haskey(_f, key) || error("key not found! [$key]") |> throw
        copy!(cache_arr_χ, _f[key])
    else
        fill!(cache_arr_χ, zero(T))

        if InitType == "Net"
            n = 20
            p = Iterators.partition(1:(N_node >> 1), N_node ÷ n) |> collect
            for I in p[2:4:end]
                cache_arr_χ[:, I] .= 1
                cache_arr_χ[:, end .- I] .= 1
                cache_arr_χ[I, :] .= 1
                cache_arr_χ[end .- I, :] .= 1
            end
        elseif InitType == "Lines"
            n = 20
            p = Iterators.partition(1:(N_node >> 1), N_node ÷ n) |> collect
            for I in p[2:4:end]
                cache_arr_χ[:, I] .= 1
                cache_arr_χ[:, end .- I] .= 1
            end

        elseif InitType == "Line"
            _n = round(Int, N_node * (1-vol) / 2)
            cache_arr_χ[:, _n+1 : end-_n] .= 1
            # cache_arr_χ[1, :] .= 1
            # cache_arr_χ[end, :] .= 1
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
    return nothing
end

"""
    initSingleSpace(::Val{Flag}, trian, a, diri_args...)
return test_space, trial_space, assembler, A, b, fe_func, ad_fe_func.
`diri_args...` is passed to `_test_and_trial_space` dirichlet.
"""
function init_single_space(::Val{F}, trian, a, diri_args...) where {F}
    test, trial, trialˢ = _test_and_trial_space(Val(F), trian, diri_args...)
    assem = SparseMatrixAssembler(trial, test)
    cell_dof_ids = get_cell_dof_ids(trial);
    du = get_trial_fe_basis(trial)
    dv = get_fe_basis(test)
    iwq = a(du, dv).dict[trian]
    rs = ([iwq], [cell_dof_ids], [cell_dof_ids])
    b = allocate_vector(assem, (nothing, [cell_dof_ids]))
    A = allocate_matrix(assem, rs)
    assemble_matrix!(A, assem, rs)
    LU = lu(A)

    n = num_free_dofs(trial)
    # `ones` for nonsingularity of `lu(A)` of heat equations
    fe_func = FEFunction(trial, ones(Float64, n))
    ad_fe_func = FEFunction(test, ones(Float64, n))
    return test, trial, trialˢ, assem, A, LU, b, fe_func, ad_fe_func
end

function _test_and_trial_space(::Val{F}, args...) where {F}
    error("flag $F not defined!") |> throw
end

function _test_and_trial_space(::Val{:StokesMini}, trian, dtags, dval)
    ref_V = LagrangianRefFE(VectorValue{2, Float64}, TRI, 1)
    V_test = TestFESpace(trian, ref_V; conformity= :H1, dirichlet_tags= dtags)
    V_trial = TrialFESpace(V_test, dval)
    Vˢ_trial = TrialFESpace(V_test, zeros(VectorValue{2, Float64}, num_dirichlet_tags(V_test)))

    ref_B = BubbleRefFE(VectorValue{2, Float64}, TRI)
    B_test = TestFESpace(trian, ref_B;)
    B_trial = TrialFESpace(B_test)

    ref_P = LagrangianRefFE(Float64, TRI, 1)
    P_test = TestFESpace(trian, ref_P; conformity= :H1, constraint= :zeromean)
    P_trial = TrialFESpace(P_test)

    trial = MultiFieldFESpace([V_trial, B_trial, P_trial])
    trialˢ = MultiFieldFESpace([Vˢ_trial, B_trial, P_trial])
    test = MultiFieldFESpace([V_test, B_test, P_test])

    return test, trial, trialˢ
end
function _test_and_trial_space(::Val{:StokesMini}, trian, Vtags, Vvals, Ptags, Pvals)
    ref_V = LagrangianRefFE(VectorValue{2, Float64}, TRI, 1)
    V_test = TestFESpace(trian, ref_V; conformity= :H1, dirichlet_tags= Vtags)
    V_trial = TrialFESpace(V_test, Vvals)
    Vˢ_trial = TrialFESpace(V_test, zeros(VectorValue{2, Float64}, num_dirichlet_tags(V_test)))

    ref_B = BubbleRefFE(VectorValue{2, Float64}, TRI)
    B_test = TestFESpace(trian, ref_B)
    B_trial = TrialFESpace(B_test)

    ref_P = LagrangianRefFE(Float64, TRI, 1)
    P_test = TestFESpace(trian, ref_P; conformity= :H1, dirichlet_tags= Ptags)
    P_trial = TrialFESpace(P_test, Pvals)
    P_trialˢ = TrialFESpace(P_test, zeros(Float64, num_dirichlet_tags(P_test)))

    trial = MultiFieldFESpace([V_trial, B_trial, P_trial])
    trialˢ = MultiFieldFESpace([Vˢ_trial, B_trial, P_trialˢ])
    test = MultiFieldFESpace([V_test, B_test, P_test])

    return test, trial, trialˢ
end

function _test_and_trial_space(::Val{:Stokes}, trian, Vtags, Vvals, Ptags, Pvals)
    ref_V = ReferenceFE(lagrangian, VectorValue{2, Float64}, 2)
    V_test = TestFESpace(trian, ref_V; conformity= :H1, dirichlet_tags= Vtags)
    V_trial = TrialFESpace(V_test, Vvals)
    V_trialˢ = TrialFESpace(V_test, zeros(VectorValue{2, Float64}, num_dirichlet_tags(V_test)))

    ref_P = LagrangianRefFE(Float64, TRI, 1)
    P_test = TestFESpace(trian, ref_P; conformity= :H1, dirichlet_tags= Ptags)
    P_trial = TrialFESpace(P_test, Pvals)
    P_trialˢ = TrialFESpace(P_test, zeros(Float64, num_dirichlet_tags(P_test)))

    trial = MultiFieldFESpace([V_trial,P_trial])
    trialˢ = MultiFieldFESpace([V_trialˢ,P_trialˢ])
    test = MultiFieldFESpace([V_test, P_test])

    return test, trial, trialˢ
end
function _test_and_trial_space(::Val{:Stokes}, trian, Vtags, Vvals)
    ref_V = ReferenceFE(lagrangian, VectorValue{2, Float64}, 2)
    V_test = TestFESpace(trian, ref_V; conformity= :H1, dirichlet_tags= Vtags)
    V_trial = TrialFESpace(V_test, Vvals)
    V_trialˢ = TrialFESpace(V_test, zeros(VectorValue{2, Float64}, num_dirichlet_tags(V_test)))

    ref_P = ReferenceFE(lagrangian, Float64, 1)
    P_test = TestFESpace(trian, ref_P; conformity= :H1, constraint= :zeromean)
    P_trial = TrialFESpace(P_test)

    trial = MultiFieldFESpace([V_trial, P_trial])
    trialˢ = MultiFieldFESpace([V_trialˢ, P_trial])
    test = MultiFieldFESpace([V_test, P_test])

    return test, trial, trialˢ
end
function _test_and_trial_space(::Val{:Heat}, trian, dtags, dval)
    ref = ReferenceFE(lagrangian, Float64, 1)
    test = TestFESpace(trian, ref; conformity= :H1, dirichlet_tags= dtags)
    trial = TrialFESpace(test, dval)
    trialˢ = TrialFESpace(test, zeros(Float64, num_dirichlet_tags(test)))
    return test, trial, trialˢ
end


"""
    solver!(x, A, fa, b)
solve `A * x = b` and store the result in `x`.
"""
@inline function solver!(x, A, fa, b)
    copy!(x, b)
    lu!(fa, A)
    ldiv!(fa, x)
    return nothing
end

"""
    smooth_funcs!(arr_fe_χ, arr_fe_χ₂, arr_fe_Gτχ, arr_fe_Gτχ₂, motion)
update `arr_fe_χ₂`, `arr_fe_Gτχ` and `arr_fe_Gτχ₂` according to `arr_fe_χ`. Note that 
this will alter corresponding `fe_χ₂`, `fe_Gτχ` and `fe_Gτχ₂`.
"""
function smooth_funcs!(arr_fe_χ, arr_fe_χ₂, arr_fe_Gτχ, arr_fe_Gτχ₂, motion) 
    @. arr_fe_χ₂ = 1 - arr_fe_χ
    motion(arr_fe_Gτχ, arr_fe_χ)
    motion(arr_fe_Gτχ₂, arr_fe_χ₂)
    return nothing
end


"""
    pde_solve!(fe_func, a, l, test, trial, A, LU, b, assem)
solve pde and update result `fe_func`.
"""
function pde_solve!(fe_func, a, l, test, trial, A, LU, b, assem)
    et = eltype(fe_func.free_values)
    fill!(fe_func.free_values, zero(et))
    uhd = fe_func

    ϕ = get_trial_fe_basis(trial)
    ψ = get_fe_basis(test)
    matcontribs = a(ϕ, ψ)
    veccontribs = l(ψ)
    data = collect_cell_matrix_and_vector(trial, test, matcontribs, veccontribs, uhd)
    assemble_matrix_and_vector!(A, b, assem, data)

    solver!(fe_func.free_values, A, LU, b)
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
    c_p = get_cell_points(trian)
    c_vals = op(c_p)
    cache_val = array_cache(c_vals)
    
    c2p = get_faces(top, Dc, 0)
    counts = zeros(eltype(c2p.ptrs), size(out)...)

    @inbounds for c_i = eachindex(c2p)
        c_val = getindex!(cache_val, c_vals, c_i)
        p_ini = c2p.ptrs[c_i]
        l = c2p.ptrs[c_i + 1] - p_ini   
        p_ini -= one(p_ini)
        for j = Base.oneto(l)
            p_i = c2p.data[p_ini + j]
            out[p_i] += c_val[j]
            counts[p_i] += one(p_ini)
        end
    end
    out ./= counts
    return nothing
end



"""
add dirichlet tag for built-in cartesian mesh.
"""
@generated function add_tag_by_filter!(m::DiscreteModel{D}, _filter, tag_name) where {D}
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

        for k = eachindex(face_to_vertices)
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

