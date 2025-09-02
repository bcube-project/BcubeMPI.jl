"""
In Paraview, use `CleanToGrid` filter to merge duplicated points
"""
function write_pvtk(
    basename::String,
    it::Int,
    time::Real,
    dmesh::DMesh{topoDim, spaceDim},
    vars::Dict{String, Tuple{V, L}};
    append = false,
    comm = MPI.COMM_WORLD,
) where {topoDim, spaceDim, V, L <: WriteVTK.AbstractFieldData}
    mypart = MPI.Comm_rank(comm) + 1
    nparts = MPI.Comm_size(comm)

    _mesh = parent(dmesh)

    # Create coordinates arrays
    vtknodes = reshape(
        [get_coords(n)[idim] for n in get_nodes(_mesh) for idim in 1:spaceDim],
        spaceDim,
        nnodes(_mesh),
    )

    # Connectivity
    c2n = Bcube.connectivities_indices(_mesh, :c2n)

    # Create cell array, and remove ghost cells
    vtkcells = [
        MeshCell(BcubeVTK.vtk_entity(Bcube.cells(_mesh)[icell]), c2n[icell]) for
        icell in dmesh.local_cells
    ]

    # Define mesh for vtk
    new_name = @sprintf("%s_%08i", basename, it)
    pvtk = pvtk_grid(
        new_name,
        vtknodes,
        vtkcells;
        part = mypart,
        nparts = nparts,
        # ghost_level = 0,
    )

    for (varname, (value, loc)) in vars
        if loc isa VTKPointData
            pvtk[varname, loc] = value
        elseif loc isa VTKCellData
            if value isa AbstractVector
                pvtk[varname, loc] = value[dmesh.local_cells]
            elseif value isa AbstractMatrix
                pvtk[varname, loc] = value[:, dmesh.local_cells]
            else
                error("value must be AbstractVector or AbstractMatrix")
            end
        else
            error("location must be VTKPointData or VTKCellData")
        end
    end

    # Append ghost info
    # ghost_nodes = zeros(Int, nnodes(_mesh))
    # ghost_cells = zeros(Int, ncells(_mesh))
    # ghost_nodes[dmesh.ghost_nodes] .= 1
    # ghost_cells[dmesh.ghost_cells] .= 1
    # pvtk["vtkGhostType", VTKPointData()] = ghost_nodes
    # pvtk["vtkGhostType", VTKCellData()] = ghost_cells

    if mypart == 1
        pvd = paraview_collection(basename; append = append)
        pvd[float(time)] = pvtk
        vtk_save(pvd)
    else
        vtk_save(pvtk)
    end
end
