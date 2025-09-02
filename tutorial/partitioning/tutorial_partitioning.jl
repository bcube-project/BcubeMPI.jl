module TutorialPartitioning

include("run_mpi.jl")
run_mpi(; nprocs = 3, filename = "driver_partitioning.jl")

end # module
