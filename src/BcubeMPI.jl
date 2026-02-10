module BcubeMPI

using Base
using Metis
using Graphs
using Bcube
using MPI
using MPIUtils
using HauntedArrays
using Printf
using WriteVTK
using SparseArrays

# This is temporary, the content of `gmsh_utils.jl` should be moved
# to BcubeGmsh
using BcubeGmsh

# This is temporary, the content of `vtk.jl` should be moved to BcubeVTK
using BcubeVTK

const MASTER = 1

# Alias unexported names from Bcube (could use `eval` here...)
const AbstractFESpace = Bcube.AbstractFESpace
const AbstractSingleFESpace = Bcube.AbstractSingleFESpace
const AbstractFunctionSpace = Bcube.AbstractFunctionSpace
const SingleFESpace = Bcube.SingleFESpace
const DofHandler = Bcube.DofHandler
const SingleFieldFEFunction = Bcube.SingleFieldFEFunction
const AbstractShape = Bcube.AbstractShape

include("./mesh/mesh.jl")
export partitioning, DistributedMesh

include("./interpolation/fespace.jl")
export allocate_array

include("./interpolation/fefunction.jl")

include("./mesh/gmsh_utils.jl")
export read_ghosts, read_partitions, read_partitioned_msh

include("parallel_factory.jl")

include("writers/vtk.jl")

end # module
