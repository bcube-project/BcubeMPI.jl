module test
using MPI
using MPIUtils
using Bcube
using BcubeMPI

MPI.Initialized() || MPI.Init()
comm = MPI.COMM_WORLD
np = MPI.Comm_size(comm)
@assert np ∈ (1, 2, 3) "Tests only valid on 1, 2 or 3 procs"

comm = MPI.COMM_WORLD

mesh_path = joinpath(@__DIR__, "..", "myout", "mesh.msh")

function test1()
    @only_root Bcube.gen_rectangle_mesh(
        mesh_path,
        :quad;
        nx = 3,
        ny = 3,
        n_partitions = np,
        split_files = true,
        create_ghosts = true,
    )
    MPI.Barrier(comm)
    mesh = read_partitioned_msh(mesh_path, comm)
    node_l2g = Bcube.absolute_indices(parent(mesh), :node)

    @one_at_a_time begin
        @show issetequal(node_l2g[Bcube.boundary_nodes(parent(mesh), 1)], [1, 2, 5])
        @show issetequal(node_l2g[Bcube.boundary_nodes(parent(mesh), 2)], [2, 3, 6])
        @show issetequal(node_l2g[Bcube.boundary_nodes(parent(mesh), 3)], [3, 4, 7])
        @show issetequal(node_l2g[Bcube.boundary_nodes(parent(mesh), 4)], [1, 4, 8])
    end
end

function test2()
    lx = 1
    ly = 2
    lz = 3
    @only_root Bcube.gen_hexa_mesh(
        "mesh.msh",
        :tetra;
        nx = 2,
        ny = 2,
        nz = 2,
        lx,
        ly,
        lz,
        xc = lx / 2,
        yc = ly / 2,
        zc = lz / 2,
        transfinite_lines = true,
        transfinite = false,
        n_partitions = np,
        split_files = true,
        create_ghosts = true,
    )
    MPI.Barrier(comm)
    mesh = read_partitioned_msh(mesh_path, comm)

    node_l2g = Bcube.absolute_indices(parent(mesh), :node)
    @one_at_a_time begin
        @show issetequal(node_l2g[Bcube.boundary_nodes(parent(mesh), 1)], [1, 4, 5, 8])
        @show issetequal(node_l2g[Bcube.boundary_nodes(parent(mesh), 2)], [2, 3, 6, 7])
        @show issetequal(node_l2g[Bcube.boundary_nodes(parent(mesh), 3)], [1, 2, 5, 6])
        @show issetequal(node_l2g[Bcube.boundary_nodes(parent(mesh), 4)], [3, 4, 7, 8])
        @show issetequal(node_l2g[Bcube.boundary_nodes(parent(mesh), 5)], [1, 2, 3, 4, 9])
        @show issetequal(node_l2g[Bcube.boundary_nodes(parent(mesh), 6)], [5, 6, 7, 8, 10])
    end
end

test1()
test2()

isinteractive() || MPI.Finalize()

end
