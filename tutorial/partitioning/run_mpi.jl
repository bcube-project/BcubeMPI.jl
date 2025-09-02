using MPI

function run_mpi(; nprocs = 1, filename)
    dir = @__DIR__
    repodir = joinpath(dir, ".")
    println(repodir)
    mpiexec() do cmd
        run(`$cmd -n $nprocs $(Base.julia_cmd()) --project=$repodir $filename $nprocs`)
    end
end
