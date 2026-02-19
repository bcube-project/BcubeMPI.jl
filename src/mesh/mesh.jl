"""
To be improved / redefined : all the attributes are not necessary

Remark : identifiying ghost nodes is not trivial and requires several exhanges iterations
between partitions. Since we don't need this info for now, it is not computed.
"""
struct DistributedMesh{topoDim, spaceDim, M} <: Bcube.AbstractMesh{topoDim, spaceDim}
    comm::MPI.Comm
    mesh::M
    ghost_tag2part::Dict{Int, Int} # ghost-cell tag (=absolute id) => partition owning that ghost
    local_cells::Vector{Int} # Index of "local" (i.e handled by local partition) cells in mesh
    ghost_cells::Vector{Int} # Index of "ghost" (i.e handled by another partition) cells in mesh
end

const DMesh = DistributedMesh

@inline get_comm(dmesh::DMesh) = dmesh.comm

Base.parent(dmesh::DMesh) = dmesh.mesh

Bcube.get_nodes(dmesh::DMesh) = Bcube.get_nodes(parent(dmesh))

function DistributedMesh(mesh::Bcube.Mesh, ghost_tag2part::Dict{Int, Int}, comm::MPI.Comm)
    ghost_tags = keys(ghost_tag2part)

    # Allocate for local and ghost indices
    n_ghosts = length(ghost_tags)
    local_cells = zeros(Int, ncells(mesh) - n_ghosts)
    ghost_cells = zeros(Int, n_ghosts)
    cell_tags = Bcube.get_absolute_cell_indices(mesh)

    # Identify local and ghost cells
    # Rq : can't we improve this loop by looping over ghost_tags instead of ncells(mesh)?
    i_local = 0
    i_ghost = 0
    for icell in 1:ncells(mesh)
        tag = cell_tags[icell]
        if tag ∈ ghost_tags # bad, to be improved
            i_ghost += 1
            ghost_cells[i_ghost] = icell
        else
            i_local += 1
            local_cells[i_local] = icell
        end
    end

    # Check
    @assert i_local == length(local_cells) "Wrong number of local cells: expected $(length(local_cells)), found $(i_local)"
    @assert i_ghost == length(ghost_cells) "Wrong number of ghost cells: expected $(length(ghost_cells)), found $(i_ghost)"

    # Tag ghost nodes
    mesh = tag_ghost_nodes(mesh, ghost_tag2part, ghost_cells, comm)

    DistributedMesh{Bcube.topodim(mesh), Bcube.spacedim(mesh), typeof(mesh)}(
        comm,
        mesh,
        ghost_tag2part,
        local_cells,
        ghost_cells,
    )
end

"""
    remove_isolated_nodes(mesh::Bcube.Mesh)

Remove nodes that are not connected to any cell.

Why ? With the current implementation of Bcube._read_msh, a lot of dummy nodes are
read for partitionned msh files. By "dummy" I mean nodes that are not connected to any cell.

This function could go in Bcube since it's not specific to MPI / BcubeMPI
"""
function remove_isolated_nodes(mesh::Bcube.Mesh)
    # tag non-isolated and isolated nodes
    isRealNode = fill(false, nnodes(mesh))
    c2n = Bcube.connectivities_indices(mesh, :c2n)
    for inodes in c2n
        isRealNode[inodes] .= true
    end
    indRealNodes = findall(isRealNode)

    # new numbering with only non-isolated nodes
    old2new = zeros(Int, nnodes(mesh))
    for (i, j) in enumerate(indRealNodes)
        old2new[j] = i
    end

    # new objects
    _nodes = get_nodes(mesh)[indRealNodes]
    _c2n = [old2new[inodes] for inodes in c2n]
    _conn = Bcube.Connectivity([nnodes(e) for e in Bcube.cells(mesh)], vcat(_c2n...))
    _bcnames = Dict(
        boundary_tag(mesh, name) => String(name) for
        (i, name) in enumerate(Bcube.boundary_names(mesh))
    )
    _bc_nodes = Dict(
        boundary_tag(mesh, name) => old2new[inodes] for
        (name, inodes) in pairs(Bcube.boundary_nodes(mesh))
    )

    _mesh = Bcube.Mesh(
        _nodes,
        Bcube.cells(mesh),
        _conn;
        bc_names = _bcnames,
        bc_nodes = _bc_nodes,
        absoluteCellIndices = Bcube.get_absolute_cell_indices(mesh),
        absoluteNodeIndices = Bcube.get_absolute_node_indices(mesh)[indRealNodes],
        metadata = Bcube.get_metadata(mesh),
    )

    return _mesh
end

"""
Nodes belonging to a ghost cell (even if shared by the current partition) are not tagged
if they lie on a boundary. This function fixes this.
"""
function tag_ghost_nodes(mesh::Bcube.Mesh, ghost_tag2part, ghost_cells, comm::MPI.Comm)
    # Alias
    mypart = MPI.Comm_rank(comm) + 1
    np = MPI.Comm_size(comm)
    n_bnd = length(Bcube.boundary_names(mesh))

    # Global numbering
    node_l2g = Bcube.get_absolute_node_indices(mesh)
    cell_l2g = Bcube.get_absolute_cell_indices(mesh)
    @show mypart, extrema(node_l2g)

    # Connectivities
    c2n = Bcube.connectivities_indices(mesh, :c2n)
    # c2c = Bcube.connectivities_indices(mesh, :c2c)

    # Determine max value of node global numbering
    l2g_max = MPI.Allreduce(maximum(node_l2g), MPI.MAX, comm)

    # Handy array
    cell2isGhost = fill(false, ncells(mesh))
    cell2isGhost[ghost_cells] .= true

    # Filter to obtain only nodes near a boundary, i.e belonging to a ghost cell or to an owned cell in contact
    # with a ghost cell.
    # node2isGhost = fill(false, nnodes(mesh))
    # for icell_l in ghost_cells
    #     node2isGhost[c2n[icell_l]] .= true
    #     for jcell_l in c2c[icell_l]
    #         node2isGhost[c2n[jcell_l]] .= true
    #     end
    # end

    # Create the node to partition full connectivity (a node can belong to several partitions)
    node2part = spzeros(Bool, nnodes(mesh), np)
    node2part[:, mypart] .= true
    for icell_l in ghost_cells
        icell_g = cell_l2g[icell_l]
        part = ghost_tag2part[icell_g]
        node2part[c2n[icell_l], part] .= true
    end

    # Create the node to bnd full connectivity (a node can belong to several boundaries)
    # We create a matrix with one row for each node. The number of columns is the number of boundary conditions.
    # For each row (= each node), we indicate if the node belongs to each the boundary condition with a Bool
    # We sort the array just in case the bnd cdts may not be sorted the same way on each partition
    tag2nodes = Bcube.boundary_nodes(mesh)
    bnd_tags = collect(keys(tag2nodes))
    Iperm = sortperm(bnd_tags)
    sorted_bnd_tags = bnd_tags[Iperm]
    n_bnd = length(sorted_bnd_tags)
    node2bnd = spzeros(Bool, nnodes(mesh), n_bnd)
    for (i, tag) in enumerate(sorted_bnd_tags)
        node2bnd[tag2nodes[tag], i] .= true
    end

    # Tricky part: we need to gather all columns into on number to ease the exchanges.
    # I decide to interpret the columns as the digits of a number written in base 2.
    # So I convert this base-2 number into base 10
    node2bnd_base2 = zeros(Int, nnodes(mesh))
    for (inode, col) in enumerate(eachrow(node2bnd))
        for (exponent, value) in enumerate(col)
            node2bnd_base2[inode] += Int(value) * 2^(exponent - 1)
        end
    end

    # Now, we transform everything into vectors to obtain lid2gid, lid2part and vals to exchange.
    # For lid2gid, we multiply the absolute numbering by the concerned partition number (to obtain a new global numbering)
    I, J, _ = findnz(node2part)
    lid2part = J
    vals = node2bnd_base2[I]
    lid2gid = (J .- 1) .* l2g_max .+ node_l2g[I]

    # Build HauntedVector
    x = HauntedVector(comm, lid2gid, lid2part, eltype(vals))
    parent(x) .= vals

    # Iterate
    x0 = copy(parent(x))
    np = MPI.Comm_size(comm)
    converged = false
    for i in 1:np
        @only_root println("Boundary node identification -> iteration $i/$np")
        HauntedArrays.update_ghosts!(x)

        # local "allgather" operation to merge all partitions contributions
        # Bitwise "or" :
        # * if a ghost find a new tag for a node we add it (i.e corresponding bit to 1)
        # * if a ghost find not tag but the curr partition has a tag, we conserve it
        # parent(x) .= parent(x) .| x0
        for (inode, value) in zip(I, parent(x))
            node2bnd_base2[inode] = node2bnd_base2[inode] | value
        end
        parent(x) .= node2bnd_base2[I]

        # Check if something has changed. Note that x can only increase (or keep constant)
        # since we perform an "or"
        _converged = sum(parent(x) .- x0) == 0
        converged = Bool(MPI.Allreduce(_converged, MPI.LAND, comm))
        if converged
            @only_root println("Convergence reached!")
            break
        else
            x0 = copy(parent(x))
        end
    end
    @assert converged "Iterations to identify boundary nodes did not converge"

    # Extract information (~reshape + from base-10 to base-2)
    node2bnd = spzeros(Bool, nnodes(mesh), n_bnd)
    for (k, (inode, value)) in enumerate(zip(I, parent(x)))
        array = Bool.(digits(value; base = 2))
        for (jbnd, b) in enumerate(array)
            node2bnd[inode, jbnd] = node2bnd[inode, jbnd] || b
        end
    end
    # display(node2bnd)

    # Now, rebuild the dict of boundary nodes
    _bcnames = Dict(
        boundary_tag(mesh, name) => String(name) for name in Bcube.boundary_names(mesh)
    )

    _bc_nodes = Dict(
        boundary_tag(mesh, name) => findall(node2bnd[:, i]) for
        (i, name) in enumerate(sorted_bnd_tags)
    )
    # @one_at_a_time @show _bc_nodes

    # Create new mesh (necessary to recreate bnd_faces etc)
    # Rq : we could do it without re-creating a Mesh, but it's simpler to process like this for now
    _conn = Bcube.Connectivity([nnodes(e) for e in Bcube.cells(mesh)], vcat(c2n...))

    _mesh = Bcube.Mesh(
        get_nodes(mesh),
        Bcube.cells(mesh),
        _conn;
        bc_names = _bcnames,
        bc_nodes = _bc_nodes,
        absoluteCellIndices = Bcube.get_absolute_cell_indices(mesh),
        absoluteNodeIndices = Bcube.get_absolute_node_indices(mesh),
    )

    return _mesh
end

"""
Nodes belonging to a ghost cell and not shared by the current partition (i.e "true" ghost node) are not tagged
if they lie on a boundary. This function fixes this.
"""
function tag_ghost_nodes_v1(mesh::Bcube.Mesh, ghost_tag2part, ghost_cells, comm::MPI.Comm)
    # Alias
    mypart = MPI.Comm_rank(comm) + 1
    np = MPI.Comm_size(comm)

    node_l2g = Bcube.get_absolute_node_indices(mesh)
    cell_l2g = Bcube.get_absolute_cell_indices(mesh)

    # @one_at_a_time @show ghost_tag2part

    # We have to build a "node to part" array. This array will be true only for owned nodes
    # but may be inexact for ghost nodes because we will assume that it belongs to the corresponding
    # ghost cell whereas it's not sure. That's why we need several iterations later on.
    # node2isGhost = fill(false, nnodes(mesh))
    # node2isGhost[ghost_nodes] .= true
    cell2isGhost = fill(false, ncells(mesh))
    cell2isGhost[ghost_cells] .= true
    # node2part = fill(mypart, nnodes(mesh))
    node2part = fill(np + 1, nnodes(mesh))
    c2n = Bcube.connectivities_indices(mesh, :c2n)
    n2c, k = Bcube.inverse_connectivity(c2n)
    # @one_at_a_time @show n2c
    # @one_at_a_time @show k
    for (inode_l, icells_l) in zip(k, n2c)
        # Loop over the cells (owned and/or ghosts) sharing this node
        # @only_root println("---- inode = $(inode_l)")
        for icell_l in icells_l
            # @only_root println("cell_j = $(icell_l)")
            if cell2isGhost[icell_l]
                # If the cell is a ghost, we try to assign to the corresponding partition to it,
                # but only if it does not belong to another partition with a lesser rank
                jcell_g = cell_l2g[icell_l]
                node2part[inode_l] = min(node2part[inode_l], ghost_tag2part[jcell_g])
                # @only_root println("is ghost! node2part = $(node2part[inode_l])")
            else
                # If the cell is not a ghost, we try to assign "mypart" to it, but only
                # if it does not belong to another partition with a lesser rank
                node2part[inode_l] = min(node2part[inode_l], mypart)
                # @only_root println("not ghost node2part = $(node2part[inode_l])")
            end
        end
    end
    @assert maximum(node2part) <= np
    # end
    # @one_at_a_time @show node2part

    # We create a matrix with one row for each node. The number of columns is the number of boundary conditions.
    # For each row (= each node), we indicate if the node belongs to each the boundary condition with a Bool
    tag2nodes = Bcube.boundary_nodes(mesh)
    # @one_at_a_time @show tag2nodes
    bnd_tags = collect(keys(tag2nodes))
    I = sortperm(bnd_tags)
    sorted_bnd_tags = bnd_tags[I]
    n_bnd = length(sorted_bnd_tags)
    node2bnd = fill(false, (nnodes(mesh), n_bnd))
    for (i, tag) in enumerate(sorted_bnd_tags)
        node2bnd[tag2nodes[tag], i] .= true
    end
    # @one_at_a_time @show Bcube.boundary_nodes(mesh)
    # @one_at_a_time display(node2bnd)

    # Tricky part: we need to gather all columns into on number to ease the exchanges.
    # I decide to interpret the columns as the digits of a number written in base 2.
    # So I convert this base-2 number into base 10
    node2bnd_base2 = zeros(Int, nnodes(mesh))
    for (inode, col) in enumerate(eachrow(node2bnd))
        for (exponent, value) in enumerate(col)
            node2bnd_base2[inode] += Int(value) * 2^(exponent - 1)
        end
    end
    # @one_at_a_time @show node2bnd_base2

    x = HauntedVector(comm, node_l2g, node2part, eltype(node2bnd_base2))
    parent(x) .= node2bnd_base2

    # Iterate
    x0 = copy(parent(x))
    np = MPI.Comm_size(comm)
    converged = false
    for i in 1:np
        @only_root println("Boundary node identification -> iteration $i/$np")
        HauntedArrays.update_ghosts!(x)
        _converged = sum(parent(x) .- x0) == 0
        converged = Bool(MPI.Allreduce(_converged, MPI.LAND, comm))
        # @one_at_a_time display(reshape(parent(x), :, n_bnd))
        if converged
            @only_root println("Convergence reached!")
            break
        else
            x0 = copy(parent(x))
        end
    end
    @assert converged "Iterations to identify boundary nodes did not converge"

    # Reshape : from base-10 to base-2
    node2bnd = fill(false, (nnodes(mesh), n_bnd))
    for (inode, value) in enumerate(parent(x))
        array = digits(value; base = 2)
        node2bnd[inode, 1:length(array)] .= Bool.(array)
    end

    # Now, rebuild the dict of boundary nodes
    _bc_nodes = Dict(tag => findall(node2bnd[:, I[i]]) for (i, tag) in enumerate(bnd_tags))
    # @one_at_a_time @show _bc_nodes

    # Create new mesh (necessary to recreate bnd_faces etc)
    # Rq : we could do it without re-creating a Mesh, but it's simpler to process like this for now
    _conn = Bcube.Connectivity([nnodes(e) for e in Bcube.cells(mesh)], vcat(c2n...))
    _absolute_node_indices = Bcube.get_absolute_node_indices(mesh)
    _absolute_cell_indices = Bcube.get_absolute_cell_indices(mesh)
    _mesh = Bcube.Mesh(
        get_nodes(mesh),
        Bcube.cells(mesh),
        _conn;
        bc_names = Bcube.boundary_names(mesh),
        bc_nodes = _bc_nodes,
    )
    Bcube.add_absolute_indices!(_mesh, :node, _absolute_node_indices)
    Bcube.add_absolute_indices!(_mesh, :cell, _absolute_cell_indices)

    return _mesh
end

"""
Nodes belonging to a ghost cell (even if shared by the current partition) are not tagged
if they lie on a boundary. This function fixes this.
"""
function tag_ghost_nodes_v2(mesh::Bcube.Mesh, ghost_tag2part, ghost_cells, comm::MPI.Comm)
    # Alias
    mypart = MPI.Comm_rank(comm) + 1
    np = MPI.Comm_size(comm)

    node_l2g = Bcube.get_absolute_node_indices(mesh)
    cell_l2g = Bcube.get_absolute_cell_indices(mesh)

    # We need to identify nodes belonging to ghost cells (even those who are also belonging to an owned cell).
    # Then we will need to assign a remote partition to each ghost node.
    # We actually do both in "one step", by tagging all ghost nodes with a partition number different from the
    # current partition number.
    # Note that `node2part` doesn't mean that the ith node is "owned" (parallel meaning) to node2part[ith],
    # but only that we'll try to receive infos from this partition.
    cell2isGhost = fill(false, ncells(mesh))
    cell2isGhost[ghost_cells] .= true
    node2part = fill(np + 1, nnodes(mesh))
    c2n = Bcube.connectivities_indices(mesh, :c2n)
    for icell_l in ghost_cells
        icell_g = cell_l2g[icell_l]
        part = ghost_tag2part[icell_g]
        for inode_l in c2n[icell_l]
            # ERROR below : imagine that the node is shared by partitions 1, 2 and 3, and that only partition 3
            # has the right info : proc1 will set node2part to '2' and proc2 will set node2part to '1' and the info
            # will never be received
            error("see comment")
            node2part[inode_l] = min(node2part[inode_l], part)
        end
    end
    node2part[findall(x -> x > np, node2part)] .= mypart
    @one_at_a_time @show node2part
    println("")

    # We create a matrix with one row for each node. The number of columns is the number of boundary conditions.
    # For each row (= each node), we indicate if the node belongs to each the boundary condition with a Bool
    tag2nodes = Bcube.boundary_nodes(mesh)
    # @one_at_a_time @show tag2nodes
    bnd_tags = collect(keys(tag2nodes))
    I = sortperm(bnd_tags)
    sorted_bnd_tags = bnd_tags[I]
    n_bnd = length(sorted_bnd_tags)
    node2bnd = fill(false, (nnodes(mesh), n_bnd))
    for (i, tag) in enumerate(sorted_bnd_tags)
        node2bnd[tag2nodes[tag], i] .= true
    end

    # Tricky part: we need to gather all columns into on number to ease the exchanges.
    # I decide to interpret the columns as the digits of a number written in base 2.
    # So I convert this base-2 number into base 10
    node2bnd_base2 = zeros(Int, nnodes(mesh))
    for (inode, col) in enumerate(eachrow(node2bnd))
        for (exponent, value) in enumerate(col)
            node2bnd_base2[inode] += Int(value) * 2^(exponent - 1)
        end
    end
    # @one_at_a_time @show node2bnd_base2

    x = HauntedVector(comm, node_l2g, node2part, eltype(node2bnd_base2))
    parent(x) .= node2bnd_base2

    # Iterate
    x0 = copy(parent(x))
    np = MPI.Comm_size(comm)
    converged = false
    for i in 1:np
        @only_root println("Boundary node identification -> iteration $i/$np")
        HauntedArrays.update_ghosts!(x)

        # Bitwise "or" :
        # * if a ghost find a new tag for a node we add it (i.e corresponding bit to 1)
        # * if a ghost find not tag but the curr partition has a tag, we conserve it
        parent(x) .= parent(x) .| x0

        # Check if something has changed. Note that x can only increase (or keep constant)
        # since we perform an "or"
        _converged = sum(parent(x) .- x0) == 0
        converged = Bool(MPI.Allreduce(_converged, MPI.LAND, comm))
        # @one_at_a_time display(reshape(parent(x), :, n_bnd))
        if converged
            @only_root println("Convergence reached!")
            break
        else
            x0 = copy(parent(x))
        end
    end
    @assert converged "Iterations to identify boundary nodes did not converge"

    # Reshape : from base-10 to base-2
    node2bnd = fill(false, (nnodes(mesh), n_bnd))
    for (inode, value) in enumerate(parent(x))
        array = digits(value; base = 2)
        node2bnd[inode, 1:length(array)] .= Bool.(array)
    end

    # Now, rebuild the dict of boundary nodes
    _bc_nodes = Dict(tag => findall(node2bnd[:, I[i]]) for (i, tag) in enumerate(bnd_tags))
    # @one_at_a_time @show _bc_nodes

    # Create new mesh (necessary to recreate bnd_faces etc)
    # Rq : we could do it without re-creating a Mesh, but it's simpler to process like this for now
    _conn = Bcube.Connectivity([nnodes(e) for e in Bcube.cells(mesh)], vcat(c2n...))
    _absolute_node_indices = Bcube.get_absolute_node_indices(mesh)
    _absolute_cell_indices = Bcube.get_absolute_cell_indices(mesh)
    _mesh = Bcube.Mesh(
        get_nodes(mesh),
        Bcube.cells(mesh),
        _conn;
        bc_names = Bcube.boundary_names(mesh),
        bc_nodes = _bc_nodes,
        absol,
    )
    Bcube.add_absolute_indices!(_mesh, :node, _absolute_node_indices)
    Bcube.add_absolute_indices!(_mesh, :cell, _absolute_cell_indices)

    return _mesh
end

# Decorator
#@inline Bcube.absolute_indices(dmesh::DistributedMesh,e::Symbol) = absolute_indices(dmesh.mesh,e) # ghost included, should we exclude?

# @inline Bcube.InteriorFaceDomain(dmesh::DMesh) = InteriorFaceDomain(parent(dmesh))
# function Bcube.BoundaryFaceDomain(dmesh::DMesh, labels::Tuple{String, Vararg{String}})
#     BoundaryFaceDomain(parent(dmesh), labels)
# end

# Enhance
# #Bcube.CellDomain(dmesh::DistributedMesh) = CellDomain(dmesh.mesh, dmesh.local_cells) #, ghosts excluded: OK for discontinuous, NOK for continuous
# Bcube.CellDomain(dmesh::DMesh) = CellDomain(parent(dmesh), 1:ncells(parent(dmesh))) # ghosts included: OK for continuous, OK for discontinuous

function partitioning(globalmesh, nparts)
    c2c = connectivity_cell2cell(globalmesh)
    graph = SimpleGraph(length(c2c))
    for (i, c2cᵢ) in enumerate(c2c)
        for j in c2cᵢ
            add_edge!(graph, i, j)
        end
    end
    return Metis.partition(graph, nparts; alg = :RECURSIVE)     #(:RECURSIVE, :KWAY)
end
