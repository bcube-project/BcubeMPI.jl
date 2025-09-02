function eigenvectors2VTK(
    basename::String,
    dsys::DistributedSystem,
    eps::SlepcEPS,
    ivecs = 1:neigs(eps);
    synchronize = false,
)
    comm = dsys.comm
    rank = MPI.Comm_rank(comm)
    dict_vars = Dict{String, Tuple{Any, WriteVTK.AbstractFieldData}}()
    iglob_min = minimum(dsys.loc2glob[dsys.localdofs])

    # Get local size
    A, B = EPSGetOperators(eps)
    irows = get_urange(A)
    nrows_l = length(irows)
    @assert nrows_l == length(get_urange(B)) # necessary condition

    # Allocate vectors
    vecr = create_vector(PETSC_DECIDE, nrows_l; comm = eps.comm)
    veci = create_vector(PETSC_DECIDE, nrows_l; comm = eps.comm)
    set_up!.((vecr, veci))

    # Fill these matrices
    for ivec in ivecs
        # Retrieve eigenvector (real and imag)
        #vecr, veci = get_eigenvector(eps, ieig)
        EPSGetEigenvector(eps, ivec - 1, vecr, veci)

        # Convert to julia arrays
        array_r, array_ref_r = VecGetArray(vecr)
        array_i, array_ref_i = VecGetArray(veci)

        # Check
        @assert length(array_r) == length(dsys.localdofs)
        @assert length(array_i) == length(dsys.localdofs)

        # Fill variables
        for (ivar, cv) in enumerate(get_cvs(dsys))
            varname = String(get_name(cv))
            var_localdofs = dsys.var_localdofs[ivar]
            var_loc2glob = dsys.var_loc2glob[ivar]

            # Trick to handle both PetscReal and PetscComplex
            # (remember that array_i = 0 in case of PetscComplex)
            array = array_r .+ im * array_i

            cv.values .= 0.0
            for iloc in var_localdofs
                iglob = var_loc2glob[iloc]
                ipetsc_loc = iglob - iglob_min + 1
                cv.values[iloc] = array[ipetsc_loc]
            end
        end

        # Free memory
        VecRestoreArray(vecr, array_ref_r)
        VecRestoreArray(veci, array_ref_i)

        # Ghost dofs have not been assigned yet. If we don't update them,
        # the rendering in Paraview is ugly because we cannot select the
        # "good" values for overlapping dofs.
        # Rq: This is madness! The synchronize will be called nvars * nvecs!!
        # For the `nvars` part, it will be resolved when we will be able
        # to update just one var.
        synchronize && update_ghost_dofs!(dsys)

        for cv in get_cvs(dsys)
            values = var_on_nodes(cv)
            varname = String(get_name(cv))
            dict_vars["$(varname)_$(ivec)_r"] = (real.(values), VTKPointData()) # WRONG CAUSE ASSUMING NODES
            dict_vars["$(varname)_$(ivec)_i"] = (imag.(values), VTKPointData()) # WRONG CAUSE ASSUMING NODES
        end
    end

    # Free vectors
    destroy!.((vecr, veci))

    # WriteVTK
    write_vtk(basename * "_$(rank)", 0, 0, get_cvs(dsys)[1].mesh, dict_vars)
end
