module test
using BcubeParallel
using Bcube
using MPI

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm) # 0-based

mesh_path = "../input/partitioned_mesh/mesh" # no rank nor extension

# Read distributed mesh
dmesh = read_partitioned_msh(mesh_path, 2, comm) # ghosts are included

# Build system
u = CellVariable(:u, dmesh, FESpace(FunctionSpace(:Lagrange, 1), :continuous))
v = CellVariable(:v, dmesh, FESpace(FunctionSpace(:Lagrange, 2), :continuous))
w = CellVariable(:w, dmesh, FESpace(FunctionSpace(:Taylor, 1), :discontinuous))
#sys = System(u)
#sys = System((u,v))
sys = System((u, v, w))

# Compute global numbering
loc2glob, dof2part = compute_dof_global_numbering(sys, dmesh)
@one_at_a_time (@show loc2glob)
@one_at_a_time (@show dof2part)

MPI.Finalize()
end
