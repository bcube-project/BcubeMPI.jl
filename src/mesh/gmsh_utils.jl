import gmsh_jll
include(gmsh_jll.gmsh_api)
import .gmsh

"""
    read_partitioned_msh(
       mesh_path::String,
        comm::MPI.Comm,
        spaceDim::Int = 0;
        verbose::Bool = false,
        rm_isolated_nodes = true,
    )

Read a partitioned gmsh file, with ghost cells.

No rank number or extension to append to `mesh_path`, just the "basename", for instance
`mesh_path = /home/toto/square`
"""
function read_partitioned_msh(
    mesh_path::String,
    comm::MPI.Comm,
    spaceDim::Int = 0;
    verbose::Bool = false,
    rm_isolated_nodes = true,
)
    _mesh_path = endswith(mesh_path, ".msh") ? replace(mesh_path, ".msh" => "") : mesh_path

    my_part = MPI.Comm_rank(comm) + 1
    full_path = _mesh_path * "_$(my_part).msh"

    @assert isfile(full_path) "Mesh file '$(full_path)' not found"

    # Read file using gmsh lib
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", Int(verbose))
    gmsh.open(full_path)

    # build local mesh
    mesh = BcubeGmsh._read_msh(spaceDim, verbose)

    # Remove isolated nodes
    if rm_isolated_nodes
        mesh = remove_isolated_nodes(mesh)
    end

    # read ghosts
    ghost_tag2part = _read_ghosts()

    # free gmsh
    gmsh.finalize()

    return DistributedMesh(mesh, ghost_tag2part, comm)
end

function read_partitioned_msh_with_cell_names(
    mesh_path::String,
    comm::MPI.Comm,
    spaceDim::Int = 0;
    verbose::Bool = false,
    rm_isolated_nodes = true,
)
    _mesh_path = endswith(mesh_path, ".msh") ? replace(mesh_path, ".msh" => "") : mesh_path

    my_part = MPI.Comm_rank(comm) + 1
    full_path = _mesh_path * "_$(my_part).msh"

    @assert isfile(full_path) "Mesh file '$(full_path)' not found"

    # Read file using gmsh lib
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", Int(verbose))
    gmsh.open(full_path)

    # build local mesh
    mesh, el_names, el_names_inv, el_cells, glo2loc_cell_indices =
        Bcube._read_msh_with_cell_names(spaceDim, verbose)

    # Remove isolated nodes
    if rm_isolated_nodes
        mesh = remove_isolated_nodes(mesh)
    end

    # read ghosts
    ghost_tag2part = _read_ghosts()

    # free gmsh
    gmsh.finalize()

    # Fix el_cells
    el_cells = tag_ghost_cells_with_physical_group(
        comm,
        mesh,
        el_cells,
        glo2loc_cell_indices,
        ghost_tag2part,
    )

    dmesh = DistributedMesh(mesh, ghost_tag2part, comm)
    return dmesh, el_names, el_names_inv, el_cells, glo2loc_cell_indices
end

"""
Version with IO, see `read_ghosts(model)` for the version where the file is already opened.
"""
function read_ghosts(path::String; verbose::Bool = false)
    # Read file using gmsh lib
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", Int(verbose))
    gmsh.open(path)

    ghost_tag2part = _read_ghosts()

    gmsh.finalize()

    return ghost_tag2part
end

"""
Applied to gmsh partitioned splitted mesh.

Find ghost elements. Return the following arrays:
- `isghost`, of size `ncells` : for each cell, says if the cell is a ghost or not
- `ghost2ind`, of size `nghosts` : the ghost -> cell local index
- `ghost2tag`, of size `nghosts` : the ghost -> cell global tag (not sure about this last one)
- `ghost2part`, of size `nghosts` : the ghost -> partition owning the ghost
"""
function _read_ghosts()
    topodim = gmsh.model.getDimension()

    # WARNING : don't build the global -> local mapping here because the order of
    # `elt_tags` is not deterministic (more specically, the order or appearance of ghost is arbitrary)
    # so it might defer from the one built by `read_msh`

    # Ghost tag (= absolute id) to partition
    ghost_tag2part = Dict{Int, Int}()

    # Find the "partition" entity tag
    entities = gmsh.model.getEntities(topodim)
    for e in entities
        # Dimension and tag of the entity:
        #topoDim = e[1]
        entity_tag = e[2]

        # Get ghosts associated to this entity (if any)
        ghosts, partitions = gmsh.model.mesh.getGhostElements(topodim, entity_tag)
        if length(ghosts) > 0
            for (g, p) in zip(ghosts, partitions)
                ghost_tag2part[g] = p
            end
            break
        end
    end

    return ghost_tag2part
end

"""
    read_partitions(path, topodim = 3; verbose = false)

Read partitions on (Bcube) entities of topology dimension `topodim`. The output is a vector whose size is the
number of entities (for instance, number of cells), and whose content is the partition number(s) owning this entity.

The order follows the order obtained by reading all gmsh elements at once (i.e not using GMSH-entities).

TODO: merge this with `read_msh` => need to determine how we store this info in `Mesh`
"""
function read_partitions(path; verbose = false)
    # Read file using gmsh lib
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", Int(verbose))
    gmsh.open(path)

    topodim = gmsh.model.getDimension()

    # Read elements, all entities mingled
    __, elt_tags, _ = gmsh.model.mesh.getElements(topodim)
    elt_tags = Int.(reduce(vcat, elt_tags))

    # Global indexing of elements is not necessarily ordered (neither dense),
    # it may be "9, 7, 8, 5, 6, 4, 3, 1, 2" for instance. So we build the
    # "elt tag" to "vector index" connectivity
    #tag2ind = invperm(elt_tags) # WRONG : indices are not dense!
    tag2ind = Dict([(tag, i) for (i, tag) in enumerate(elt_tags)])

    # Allocate a Vector of vectors : for each mesh element, a Vector a Int
    # containing the partition numbers
    # If we knew the total number of partitions, we could allocate a big 'zeros'
    # matrix...
    elt2part = Vector{Vector{Int}}(undef, length(elt_tags))

    # Read entities and loop
    entities = gmsh.model.getEntities(topodim)
    for e in entities
        # Dimension and tag of the entity:
        #topodim = e[1]
        entity_tag = e[2]

        # Check for partitions for this entity
        partitions = gmsh.model.getPartitions(topodim, entity_tag)

        # Test if this is a "partition" entity
        if length(partitions) > 0

            # Get the mesh elements for this "partition" entity (topodim, entity_tag):
            __, elt_tags, _ = gmsh.model.mesh.getElements(topodim, entity_tag)
            length(elt_tags) == 0 && continue
            elt_tags = Int.(reduce(vcat, elt_tags))

            # Convert to vector of Int
            part = convert(Vector{Int}, partitions)

            # Loop over elements in this partition entity
            for elt_tag in elt_tags
                # Let's say element tags are "3, 9, 6". Then we want to specify the partition numbers
                # for these elements. To find the right place, we use 'tag2ind'
                elt2part[tag2ind[elt_tag]] = part
            end
        end
    end

    gmsh.finalize()

    return elt2part
end

"""
    partition_msh(path, n_partitions; kwargs...)

Partition a gmsh ".msh" file into `n_partitions`.

Available kwargs are
* `verbose` : `true` or `false` to enable gmsh verbose
* `msh_format` : floating number indicating the output msh format (for instance : `2.2`)
* `split_files` : if `true`, create one file by partition
* `create_ghosts` : if `true`, add a layer of ghost cells at every partition boundary
"""
function partition_msh(path, n_partitions; kwargs...)
    # Create mesh
    gmsh.initialize()
    BcubeGmsh._apply_gmsh_options(; kwargs...)
    gmsh.open(path)
    gmsh.model.mesh.partition(n_partitions)
    gmsh.write(path)
    gmsh.finalize()
end

"""
Currently, gmsh doesn't associate any tag to ghost cell. This function fixes it
"""
function tag_ghost_cells_with_physical_group(
    comm,
    mesh,
    el_cells,
    glo2loc_cell_indices,
    ghost_tag2part,
)

    # Alias
    mypart = MPI.Comm_rank(comm) + 1

    # Global numbering
    cell_l2g = Bcube.absolute_indices(mesh, :cell)

    # Build a cell to physical-tag connectivity
    tags = collect(keys(el_cells))
    I = sortperm(tags)
    sorted_tags = tags[I]
    n_tags = length(sorted_tags)
    cell2tag = spzeros(Bool, ncells(mesh), n_tags)
    for (j, tag) in enumerate(sorted_tags)
        indices = el_cells[tag]
        for icell_g in indices
            icell_l = glo2loc_cell_indices[icell_g]
            cell2tag[icell_l, j] = true
        end
    end

    # Tricky part: we need to gather all columns into on number to ease the exchanges.
    # I decide to interpret the columns as the digits of a number written in base 2.
    # So I convert this base-2 number into base 10
    cell2tag_base2 = zeros(Int, ncells(mesh))
    for (inode, col) in enumerate(eachrow(cell2tag))
        for (exponent, value) in enumerate(col)
            cell2tag_base2[inode] += Int(value) * 2^(exponent - 1)
        end
    end

    # Prepare HauntedVector
    lid2gid = cell_l2g
    lid2part = fill(mypart, ncells(mesh))
    for (tag, part) in ghost_tag2part
        lid2part[glo2loc_cell_indices[tag]] = part
    end
    values = cell2tag_base2
    x = HauntedVector(comm, lid2gid, lid2part, eltype(values))
    parent(x) .= values

    # Exchange the info
    HauntedArrays.update_ghosts!(x)

    # Extract information (~reshape + from base-10 to base-2)
    cell2tag = spzeros(Bool, ncells(mesh), n_tags)
    for (icell_l, value) in enumerate(parent(x))
        array = Bool.(digits(value; base = 2))
        cell2tag[icell_l, 1:length(array)] .= array
    end

    # Now, rebuild the dict of el_cells
    _el_cells = Dict(
        tag => cell_l2g[findall(cell2tag[:, i])] for (i, tag) in enumerate(sorted_tags)
    )

    return _el_cells
end
