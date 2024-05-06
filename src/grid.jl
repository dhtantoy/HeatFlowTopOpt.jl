function createGrid(::Val{2}, N_cell, L, file)
    gmsh.initialize()
    gmsh.model.add("tmp")

    lc = 0.1

    gmsh.model.geo.addPoint(-1L, 1L, 0, lc, 1)
    gmsh.model.geo.addPoint(-1L, 2L, 0, lc, 2)
    gmsh.model.geo.addPoint(0, 2L, 0, lc, 3)
    gmsh.model.geo.addPoint(0, 3L, 0, lc, 4)
    gmsh.model.geo.addPoint(3L, 3L, 0, lc, 5)
    gmsh.model.geo.addPoint(3L, 2L, 0, lc, 6)
    gmsh.model.geo.addPoint(4L, 2L, 0, lc, 7)
    gmsh.model.geo.addPoint(4L, 1L, 0, lc, 8)
    gmsh.model.geo.addPoint(3L, 1L, 0, lc, 9)
    gmsh.model.geo.addPoint(3L, 0, 0, lc, 10)
    gmsh.model.geo.addPoint(0, 0, 0, lc, 11)
    gmsh.model.geo.addPoint(0, 1L, 0, lc, 12)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)
    gmsh.model.geo.addLine(7, 8, 7)
    gmsh.model.geo.addLine(8, 9, 8)
    gmsh.model.geo.addLine(9, 10, 9)
    gmsh.model.geo.addLine(10, 11, 10)
    gmsh.model.geo.addLine(11, 12, 11)
    gmsh.model.geo.addLine(12, 1, 12)
    gmsh.model.geo.addLine(3, 12, 13)
    gmsh.model.geo.addLine(6, 9, 14)

    gmsh.model.geo.addCurveLoop([1, 2, 13, 12], 15)
    gmsh.model.geo.addCurveLoop([3, 4, 5, 14, 9, 10, 11, -13], 16)
    gmsh.model.geo.addCurveLoop([-14, 6, 7, 8], 17)

    gmsh.model.geo.addPlaneSurface([15], 1)
    gmsh.model.geo.addPlaneSurface([16], 2)
    gmsh.model.geo.addPlaneSurface([17], 3)

    for i ∈ [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14]
        gmsh.model.geo.mesh.setTransfiniteCurve(i, N_cell + 1)
    end 

    for i ∈ [4, 10]
        gmsh.model.geo.mesh.setTransfiniteCurve(i, 3N_cell + 1)
    end
    gmsh.model.geo.mesh.setTransfiniteSurface(1)
    gmsh.model.geo.mesh.setTransfiniteSurface(2, "Left", [4, 5, 10, 11])
    gmsh.model.geo.mesh.setTransfiniteSurface(3)

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(2)

    # "wall" 1
    gmsh.model.addPhysicalGroup(1, [2, 3, 4, 5, 6, 8, 9, 10, 11, 12], 1) 
    # "inlet" 2
    gmsh.model.addPhysicalGroup(1, [1], 2)
    # "outlet" 3
    gmsh.model.addPhysicalGroup(1, [7], 3)
    # "block_left" 4
    gmsh.model.addPhysicalGroup(2, [1], 4)
    # "block_center" 5
    gmsh.model.addPhysicalGroup(2, [2], 5)
    # "block_right" 6
    gmsh.model.addPhysicalGroup(2, [3], 6)
    # "body" 7
    gmsh.model.addPhysicalGroup(2, [1, 2, 3], 7)

    gmsh.write(file)
    gmsh.finalize()

    return nothing
end

"""
need to be debugged.
"""
function sort_axes_lt(p::VectorValue{D}, q::VectorValue{D}, tol::Real= 1e-4) where D
    err = p - q 
    idx = findlast(x->abs(x) >= tol, err.data)
    
    return isnothing(idx) || err[idx] < zero(eltype(err)) 
end

# """
# return `perm` such that for given `χ` in cartesian grid with cartesian coordinates `coors`, 
# `χ[perm]` could be interpolated to a FESpace based on `aux_Ω` directly, e.g. f,
# with geometric structure as same as `evaluate(f, coors)`.
# """
# function perm_map_to_cartesian(aux_Ω::Triangulation)
#     coors = get_node_coordinates(aux_Ω)
#     perm = sortperm(coors; lt= sort_axes_lt)
#     return perm_map_to_cartesian(perm)
# end

# function perm_map_to_cartesian(perm::Vector)
#     n = length(perm)
#     ret = similar(perm)
#     A = Diagonal(Fill(1, n))
#     P = A[perm, :]
#     @inbounds @threads for i = axes(P, 2)
#         v = view(P, :, i)
#         idx = findfirst(!iszero, v)
#         ret[i] = idx
#     end
    
#     return ret
#     # return rowvals(sparse(P))
# end
