module test
using BcubeParallel
using Bcube
using MPI
using WriteVTK

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm) # 0-based

mesh_path = "../input/partitioned_mesh/mesh" # no rank nor extension

# Read mesh
dmesh = read_partitioned_msh(mesh_path, 2, comm) # ghosts are included

# Build variable(s)
u = CellVariable(:u, dmesh, FESpace(FunctionSpace(:Lagrange, 1), :continuous))
v = CellVariable(:v, dmesh, FESpace(FunctionSpace(:Lagrange, 2), :continuous))
w = CellVariable(:w, dmesh, FESpace(FunctionSpace(:Taylor, 1), :discontinuous))

# Assign values
set_values!(u, xy -> 100 * rank + xy[1] + xy[2])
set_values!(w, xy -> 100 * rank + xy[1] + xy[2])

# Build distributed system
#@only_root println("Building distributed system...")
#dsys = DistributedSystem(u)
#dsys = DistributedSystem((u,v))
dsys = DistributedSystem((u, v, w), dmesh) # Rq: we could have a DistributedCellVariable to encapsulate `dmesh`...

# Update values
@only_root println("Updating dofs...")
update_ghost_dofs!(dsys)

# Write results
vals_u = var_on_nodes(u)
vals_w = var_on_centers(w)
write_vtk(
    "../myout/output_$rank",
    0,
    0,
    dmesh.mesh,
    Dict("u" => (vals_u, VTKPointData()), "w" => (vals_w, VTKCellData())),
)

MPI.Finalize()
end
