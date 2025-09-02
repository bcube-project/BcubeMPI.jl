module Helmholtz #hide
println("Running Helmholtz example...") #hide
# # Helmholtz with SLEPc
# # Theory
# We consider the following Helmholtz equation, representing for instance the acoustic wave propagation with Neuman boundary condition(s):
# ```math
# \begin{cases}
#   \Delta u + \omega^2 u = 0 \\
#   \dfrac{\partial u}{\partial n} = 0 \textrm{  on  } \Gamma
# \end{cases}
# ```
#
# # How to run
# The present tutorial is devoted to find this Helmholtz eigenvalues (and eigenvectors). To run this tutorial, execute `mpirun -n 3 julia helmholtz.jl`;
# using of course any number of processors you want. The results may be visualized with Paraview.
#
# # The code
# Load the necessary packages
const dir = string(@__DIR__, "/")
using Bcube
using BcubeParallel
using LinearAlgebra
using WriteVTK
using Printf
using SparseArrays
using MPI
using PetscWrap
using SlepcWrap
include(joinpath(@__DIR__, "petsc_utils.jl"))
include(joinpath(@__DIR__, "slepc_utils.jl"))

# Settings
const write_eigenvectors = true
const out_dir = dir * "../myout/"

# Init Slepc to init MPI comm (to be improved, should be able to start Slepc from existing comm...). SLEPc arguments can be set
# here with a string, or using command line arguments.
SlepcInitialize("-eps_nev 50 -st_pc_factor_shift_type NONZERO -st_type sinvert -eps_view")

# Since MPI has been initialized by SLEPc, we can retrieve MPI infos such as the rank or the number of processors.
#MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm) + 1 # '+1' to start at 1
nprocs = MPI.Comm_size(comm)

# Generate the mesh : only one processor needs to compute the mesh, hence the use of the `@only_root` macro.
const mesh_path = out_dir * "mesh"

#- Generate a 1D mesh : (un)comment
#-spaceDim = 1; topoDim = 1
#-isRoot && gen_line_mesh(mesh_path; nx = 4, npartitions = nprocs)

#- Generate a 2D mesh : (un)comment
spaceDim = 2
topoDim = 2
@only_root gen_rectangle_mesh(
    mesh_path * ".msh",
    :tri;
    nx = 11,
    ny = 11,
    lx = 1.0,
    ly = 1.0,
    xc = 0.5,
    yc = 0.5,
    npartitions = nprocs,
    split_files = true,
    create_ghosts = true,
)

#- Generate a 3D mesh : (un)comment
#-spaceDim = 3; topoDim = 3
#-@only_root gen_cylinder_mesh(mesh_path * ".msh", 10., 30; npartitions = nprocs, split_files = true, create_ghosts = true)

# Now that the mesh is available, each processor can read its partition.
MPI.Barrier(comm) # all procs must wait for the mesh to be built (otherwise they start reading a wrong mesh)
dmesh = read_partitioned_msh(mesh_path, 2, comm) # '2' indicates the space dimension (3 by default)

# Next, create a scalar variable named `:u`. The Lagrange polynomial space is used here. By default,
# a "continuous" function space is created (by opposition to a "discontinuous" one). The order is set to `1`.
# With this variable we can create a `DistributedSystem`, which will wrap MPI communications.
const degree = 1
fs = FunctionSpace(:Lagrange, degree)
fes = FESpace(fs, :continuous; size = 1) #  size=1 for scalar variable
ϕ = CellVariable(:ϕ, dmesh.mesh, fes, ComplexF64)
dsys = DistributedSystem(ϕ, dmesh)
@one_at_a_time ndofs(ϕ)

# Create a `TestFunction`
λ = get_trial_function(ϕ)

# Define measures for cell and interior face integrations
dΩ = Measure(CellDomain(dmesh), 2) # no quadrature higher than 2 for Penta6...

# Compute volumic integrals that constitute the weak form of the equation
@only_root println("Computing integrals...")
_A = ∫(∇(λ) * transpose(∇(λ)))dΩ
_B = ∫(λ * transpose(λ))dΩ

# Build julia sparse matrices from integration result
@only_root println("Assembling...")

As = sparse(_A, ϕ)
Bs = sparse(_B, ϕ)

# Convert to SLEPc matrices
@only_root println("Converting to Petsc matrices...")
A = julia_sparse_to_petsc(As, dsys)
B = julia_sparse_to_petsc(Bs, dsys)

# Now we set up the eigenvalue solver
@only_root println("Creating EPS...")
eps = create_eps(A, B; auto_setup = true)

# Then we solve
@only_root println("Solving...")
solve!(eps)

# Retrieve eigenvalues
@only_root println("Number of converged eigenvalues : $((neigs(eps)))")
i_eigs = 1:min(50, neigs(eps))
vp = get_eig(eps, i_eigs)
@only_root (@show sqrt.(abs.(vp[3:8])))

# Display the "first" eigenvalues:
@only_root sqrt.(abs.(vp[i_eigs]))

# Write result to ascii
const casename = "helmholtz_slepc_np$(nprocs)"
@only_root println("Writing results to files")
@only_root println("Writing eigenvalues to '$(casename)'...")
eigenvalues2file(
    eps,
    out_dir * casename * "_vp.csv";
    two_cols = true,
    write_index = true,
    write_header = true,
    comment = "",
)
if write_eigenvectors
    @only_root println("Writing eigenvectors...")
    eigenvectors2VTK(out_dir * casename, dsys, eps; synchronize = true)
end

# Free memory
@only_root println("Destroying Petsc/Slepc objects")
destroy!.((A, B, eps))

# The end (the Barrier helps debugging)
println("processor $(rank)/$(nprocs) reached end of script")
MPI.Barrier(comm)
SlepcFinalize()

end #hide
