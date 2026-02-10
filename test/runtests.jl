using BcubeMPI
using Test

# using MPI
# MPI.Init()
# comm = MPI.COMM_WORLD
# print("Hello world, I am rank $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))\n")
# MPI.Barrier(comm)
# MPI.Finalize()
include("run_mpi.jl")

@testset "BcubeMPI.jl" begin
    @test 1 === 1
    run_mpi(; nprocs = 4, filename = "test_hello.jl")
end
