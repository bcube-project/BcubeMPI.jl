module test
using MPI
using MPIUtils
using Bcube
using BcubeGmsh
using BcubeMPI

MPI.Initialized() || MPI.Init()
comm = MPI.COMM_WORLD
np = MPI.Comm_size(comm)
@assert np ∈ (1, 2, 3) "Tests only valid on 1, 2 or 3 procs"

comm = MPI.COMM_WORLD

mesh_path = joinpath(@__DIR__, "..", "myout", "mesh.msh")
@only_root mkpath(dirname(mesh_path))

function test1()
    @only_root BcubeGmsh.gen_rectangle_mesh(
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
    node_l2g = Bcube.get_absolute_node_indices(parent(mesh))

    @one_at_a_time begin
        @assert issetequal(
            node_l2g[Bcube.boundary_nodes(parent(mesh), boundary_tag(mesh, "South"))],
            [1, 2, 5],
        )
        @assert issetequal(
            node_l2g[Bcube.boundary_nodes(parent(mesh), boundary_tag(mesh, "East"))],
            [2, 3, 6],
        )
        @assert issetequal(
            node_l2g[Bcube.boundary_nodes(parent(mesh), boundary_tag(mesh, "North"))],
            [3, 4, 7],
        )
        @assert issetequal(
            node_l2g[Bcube.boundary_nodes(parent(mesh), boundary_tag(mesh, "West"))],
            [1, 4, 8],
        )
    end
end

function test2()
    lx = 1
    ly = 2
    lz = 3
    @only_root BcubeGmsh.gen_hexa_mesh(
        mesh_path,
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

    node_l2g = Bcube.get_absolute_node_indices(parent(mesh))
    @one_at_a_time begin
        @assert issetequal(
            node_l2g[Bcube.boundary_nodes(parent(mesh), boundary_tag(mesh, "xmin"))],
            [1, 4, 5, 8],
        )
        @assert issetequal(
            node_l2g[Bcube.boundary_nodes(parent(mesh), boundary_tag(mesh, "xmax"))],
            [2, 3, 6, 7],
        )
        @assert issetequal(
            node_l2g[Bcube.boundary_nodes(parent(mesh), boundary_tag(mesh, "ymin"))],
            [1, 2, 5, 6],
        )
        @assert issetequal(
            node_l2g[Bcube.boundary_nodes(parent(mesh), boundary_tag(mesh, "ymax"))],
            [3, 4, 7, 8],
        )
        @assert issetequal(
            node_l2g[Bcube.boundary_nodes(parent(mesh), boundary_tag(mesh, "zmin"))],
            [1, 2, 3, 4, 9],
        )
        @assert issetequal(
            node_l2g[Bcube.boundary_nodes(parent(mesh), boundary_tag(mesh, "zmax"))],
            [5, 6, 7, 8, 10],
        )
    end
end

test1()
test2()

isinteractive() || MPI.Finalize()

end
