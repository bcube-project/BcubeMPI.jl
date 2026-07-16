using BcubeMPI
using Test

include("run_mpi.jl")

@testset "BcubeMPI.jl" begin
    @test 1 === 1 # smoke test
    run_mpi(; nprocs = 4, filename = "test_hello.jl")
    run_mpi(; nprocs = 3, filename = "test_np.jl")
    run_mpi(;
        nprocs = 3,
        filename = joinpath("..", "tutorial", "linear_transport", "linear_transport.jl"),
    )
end
