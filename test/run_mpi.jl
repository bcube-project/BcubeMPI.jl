using MPI
using Test
function run_mpi(; nprocs = 1, filename)
    dir = @__DIR__
    repodir = joinpath(dir, "..")
    mpiexec() do cmd
        run(`$cmd -n $nprocs $(Base.julia_cmd()) --project=$repodir $filename`)
        @test true
    end
end
