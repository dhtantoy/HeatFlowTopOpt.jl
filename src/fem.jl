
"""
    struct LinearOpWithCache{O, C, S, W, L}
        # *
    end
cache for solving a linear problem.
"""
struct LinearOpWithCache{O, C, S, W, L}
    aop::O
    cache::C
    assem::S
    weakform::W
    ls::L
    LinearOpWithCache(aop, cache, assem, weakform, ls) = new{typeof(aop), typeof(cache), typeof(assem), typeof(weakform), typeof(ls)}(aop, cache, assem, weakform, ls)
end

"""
    struct NLOpWithCache{O, C, N}
        # *
    end 
cache for solving a nonlinear problem.
"""
struct NLOpWithCache{O, C, N}
    op::O
    cache::C
    nls::N
    NLOpWithCache(op, cache, nls) = new{typeof(op), typeof(cache), typeof(nls)}(op, cache, nls)
end

"""
    init_chi!(arr_χ, InitType; kwargs...)
in-place initialize `cache_arr_χ` according to `InitType`.

# Arguments
- vol::Float64: volume of the initial region. default 0.4.
- seed::Int: seed for random initialization. default 0.
- file::String: file path for initialization.
- key::String: key for initialization.
"""
function init_chi!(arr_χ::Array{T}, InitType; vol= 0.4, seed= 0, file="", key="") where T
    N_node = size(arr_χ, 1)
    if InitType == "All"
        fill!(arr_χ, one(T))
    elseif InitType == "File"
        isfile(file) || error("file not found! [$file]") |> throw
        _f = load(file)
        haskey(_f, key) || error("key not found! [$key]") |> throw
        copy!(arr_χ, _f[key])
    else
        fill!(arr_χ, zero(T))

        if InitType == "Net"
            n = 20
            p = Iterators.partition(1:(N_node >> 1), N_node ÷ n) |> collect
            for I in p[2:4:end]
                arr_χ[:, I] .= 1
                arr_χ[:, end .- I] .= 1
                arr_χ[I, :] .= 1
                arr_χ[end .- I, :] .= 1
            end
        elseif InitType == "Lines"
            n = 20
            p = Iterators.partition(1:(N_node >> 1), N_node ÷ n) |> collect
            for I in p[2:4:end]
                arr_χ[:, I] .= 1
                arr_χ[:, end .- I] .= 1
            end

        elseif InitType == "Line"
            _n = round(Int, N_node * (1-vol) / 2)
            arr_χ[:, _n+1 : end-_n] .= 1
            # cache_arr_χ[1, :] .= 1
            # cache_arr_χ[end, :] .= 1
        elseif InitType == "Rand"
            m₁, m₂ = size(arr_χ)
            c::Int = ceil(m₂ / 2)
            f::Int = floor(m₂ / 2)
            N = c * m₁
            perm = randperm(Random.seed!(seed), N)
            M = round(Int, N * vol)
            arr_χ[ perm[1:M] ] .= 1

            arr_χ[:, c+1:end] = arr_χ[:, f:-1:1]
        else
            error("InitType not defined!") |> throw
        end
    end
    return nothing
end

"""
    init_solve(weakform, trial, test)
initialize the solver for linear problem.
"""
function init_solve(weakform, trial, test)
    assem = SparseMatrixAssembler(trial, test)
    op = AffineFEOperator(weakform, trial, test, assem)
    ls = LUSolver()
    fe_func, cache = solve!(zero(trial), ls, op, nothing)
    opc = LinearOpWithCache(op, cache, assem, weakform, ls)
    return opc, fe_func
end
    
"""
    init_solve(res, jac, trial, test)
initialize the solver for nonlinear problem.
"""
function init_solve(res, jac, trial, test)
    assem = SparseMatrixAssembler(trial, test)
    op = FEOperator(res, jac, trial, test, assem)
    nls = NLSolver(LUSolver(), show_trace=false, method=:newton, linesearch=BackTracking(), ftol= 1e-8)
    fe_func, cache = solve!(zero(trial), FESolver(nls), op, nothing)
    opc = NLOpWithCache(op, cache, nls)
    return opc, fe_func
end

"""
    smooth_funcs!(arr_χ, arr_χ₂, arr_Gτχ, arr_Gτχ₂, motion)
update `arr_χ₂`, `arr_Gτχ` and `arr_Gτχ₂` according to `arr_χ`. Note that 
this will alter corresponding `fe_χ₂`, `fe_Gτχ` and `fe_Gτχ₂`.
"""
function smooth_funcs!(arr_χ, arr_χ₂, arr_Gτχ, arr_Gτχ₂, motion) 
    @. arr_χ₂ = 1 - arr_χ
    motion(arr_Gτχ, arr_χ)
    motion(arr_Gτχ₂, arr_χ₂)
    return nothing
end

"""
    pde_solve!(opc::LinearOpWithCache, fe_func)
in-place solve the linear problem.
"""
function pde_solve!(opc::LinearOpWithCache, fe_func)
    x = get_free_dof_values(fe_func)
    et = eltype(x)
    fill!(x, zero(et))
    feop = opc.aop
    cache = opc.cache
    weakform = opc.weakform
    assem = opc.assem
    ls = opc.ls
    A = get_matrix(feop)
    b = get_vector(feop)
    test = get_test(feop)
    trial = get_trial(feop)

    u = get_trial_fe_basis(trial)
    v = get_fe_basis(test)
    matcontribs, veccontribs = weakform(u, v)
    data = collect_cell_matrix_and_vector(trial, test, matcontribs, veccontribs, fe_func)
    assemble_matrix_and_vector!(A, b, assem, data)
    op = get_algebraic_operator(feop)

    solve!(x, ls, op, cache)
    nothing
end
    
"""
    pde_solve!(opc::NLOpWithCache, fe_func)
in-place solve the nonlinear problem.
"""
function pde_solve!(opc::NLOpWithCache, fe_func)
    x = get_free_dof_values(fe_func)
    et = eltype(x)
    fill!(x, zero(et))
    cache = opc.cache
    feop = opc.op
    nls = opc.nls
    op = get_algebraic_operator(feop)

    f0 = cache.f0
    j0 = cache.j0
    ns = cache.ns
    f!(r,x) = residual!(r,op,x)
    j!(j,x) = jacobian!(j,op,x)
    fj!(r,j,x) = residual_and_jacobian!(r,j,op,x)
    
    residual_and_jacobian!(f0,j0,op,x)
    df = Algebra.OnceDifferentiable(f!,j!,fj!,x,f0,j0)
    numerical_setup!(ns,j0)
    cache.df = df 
    cache.result = nothing
    Algebra._nlsolve_with_updated_cache!(x,nls,op,cache)
    nothing
end


"""
    _compute_node_value!(out, op, trian)
compute the value of each node of `trian` according to the value of its adjacent cells.
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

