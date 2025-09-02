include("partitioning.jl")
nparts = parse(Int, ARGS[1])
prun(tuto_partitioning, mpi, nparts)
