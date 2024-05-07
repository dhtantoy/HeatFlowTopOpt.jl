using HeatFlowTopOpt
using Gridap 

# m = CartesianDiscreteModel((0, 1, 0, 1), (2, 2))
m = DiscreteModelFromFile("models/N_12_L_1.0.msh")

Ω = Triangulation(m)
null = Triangulation(m, [])

V = TestFESpace(m, ReferenceFE(lagrangian, VectorValue{2, Float64}, 1); conformity= :H1, dirichlet_tags= [4])

s = TestFESpace(null)
null_f = FEFunction(s)
num_free_dofs(s)

# f = FEFunction(V, 1.:10000)

