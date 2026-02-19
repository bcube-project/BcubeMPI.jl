const _N_INFOS = 2 # number of infos to identify a dof

"""
DEPRECATED DOCUMENTATION

Notes: there are multiple levels of dofs numering for a system of variables on a distributed mesh.
Consider a dof a variable, this variable being part of a system of variables.
- this dof has a local index on one or several cells.
  Ex: dof 3 of cell 21 and dof 5 of cell 897 for instance
- this dof has a local index for the variable on the local mesh, all cells considered.
  Ex: dof 456 of this variable on the local mesh
- this dof has a local index for the entire system on the local mesh.
  Ex: dof 456 of this variable is dof 15135 of the system on the local mesh
- this dof has a global index for the entire system, all partitions considered
- this dof also has a global index for the variable (...)



* identify dofs handled by local proc and dofs by another partition
* filter dofs not handled by local partition
* indicate number of ghost partitions and ghost partitions id (MPI.Allgather)
* send number of dofs "handled" by each ghost partition
* identify all ghost dofs by `ivar`, `icell_g` and `iloc` (`iloc` being the index of the dof in the cell for this variable)
* iterate:
    send them to each concerned ghost partition
    receive them from each partition
    in the received dofs, apply the "min" function on the assigned partition number
    answer with the updated dofs partition
* now we have an accurate `dof2part`
* count number of dofs handled by local partition
* gather dofs count
* compute offset from dofs count and rank
* build global DofHandler with offset, assigning random number for ghost dofs
* ask for the global number of ghost dofs to the other partition. To do so (no need to iterate):
    indicate number of ghost partitions and number of dofs handled by each ghost partition (MPI.Allgather)
    identify all ghost dofs by `ivar`, `icell_g` and `iloc` (`iloc` being the index of the dof in the cell for this variable)
    send them to each concerned ghost partition
    receive them from each partition


WARNING :
- `to be recv` designates an information that is asked by the local partition (so it's an information that will be received)
- `to be sent` designates an information that is provided by the local partition (so it's an information that will be sent)
"""

"""
Compute the dof global numbering from the `DofHandler` and the partitioned mesh. MPI must
have been initialized before calling this function.
"""
function compute_dof_global_numbering(dhl::DofHandler, dmesh::DistributedMesh)
    comm = dmesh.comm
    rank = MPI.Comm_rank(comm) # 0-based
    nprocs = MPI.Comm_size(comm)
    nparts = nprocs
    my_part = rank + 1 # local partition number (1-based)

    # Mesh numbering
    mesh = dmesh.mesh
    ind2tag = Bcube.get_absolute_cell_indices(mesh) # global id of cells, ghost included
    tag2ind = Dict{Int, Int}(tag => ind for (ind, tag) in enumerate(ind2tag))
    #@one_at_a_time ((get_nvars(sys) == 1) && (@show dof2coords(sys, mesh))) # debug print

    # Retrieve absolute indices of ghosts
    ghost_tag2part = dmesh.ghost_tag2part
    #nghosts = length(keys(ghost_tag2part))

    # Retrieve local indices of ghosts
    # Warning : don't merge this into `read_ghosts` because of arbitrary ordering of `elt_tags`
    ghost_tag2indpart = Dict{Int, NTuple{2, Int}}(
        tag => (findfirst(==(tag), ind2tag), part) for (tag, part) in ghost_tag2part
    ) # (icell_glob => (icell_loc, partition))

    # Compute local dof index -> partition handling this dof (at this point, only true for dofs handled by current partition)
    dof2part = _compute_dof2part(ghost_tag2part, dmesh.mesh, dhl, my_part, nparts)

    # Filter to obtain only ghost-dof -> part
    ghostdof2part = Dict{Int, Int}()
    for (idof_l, ipart) in enumerate(dof2part)
        if ipart != my_part
            ghostdof2part[idof_l] = ipart
        end
    end
    ghost_parts = unique(values(ghostdof2part))
    #@one_at_a_time (@show ghost_parts)

    # Identify partitions that are sending infos to local partition
    src_parts = _identify_src_partitions(ghost_parts, comm)
    #@one_at_a_time (@show src_parts)

    # Count number of dofs that are sent to the local partition by each `src` partition
    # (and also send this info from local part to `dest` parts)
    tobesent_part2ndofs, _ =
        _identify_src_ndofs(ghostdof2part, ghost_parts, src_parts, comm)
    #@one_at_a_time (@show tobesent_part2ndofs)

    # Build information identifying ghost dofs to be sent and received
    toberecv_part2dofsid, toberecv_part_to_idof_l = _build_send_ghost_dofs_identifiers(
        dof2part,
        ghostdof2part,
        ghost_parts,
        ghost_tag2indpart,
        dhl,
    )
    tobesent_part2dofsid = Dict{Int, Matrix{Int}}(
        ipart => zeros(Int, tobesent_part2ndofs[ipart], _N_INFOS) for ipart in src_parts
    )
    #@one_at_a_time (@show toberecv_part2dofsid)
    #@one_at_a_time (@show toberecv_part_to_idof_l)

    # Identify local dofs asked by `src` partitions
    tobesent_part_to_idof_l =
        _identify_asked_dofs(toberecv_part2dofsid, tobesent_part2dofsid, dhl, tag2ind, comm)
    #@one_at_a_time (@show tobesent_part_to_idof_l)

    # Iterate until convergence
    #@one_at_a_time (@show dof2part)
    converged = false
    for _ in 1:nprocs
        @only_root println("new iteration -> dof2part") comm
        converged = _update_dof2part!(
            dof2part,
            tobesent_part_to_idof_l,
            toberecv_part_to_idof_l,
            comm,
        )
        converged && break
        #@one_at_a_time (@show dof2part)
    end
    @assert converged "Iterations to determine dof2part are not converged"

    # Count number of dofs handle by the local partition
    nd_loc_inner = count(==(my_part), dof2part)

    # Gather all dofs counts
    all_nd = MPI.Allgather(nd_loc_inner, comm)
    #@only_root (@show all_nd)

    # Compute offset for the global numbering of local dofs
    offset = sum(all_nd[1:rank])

    # Apply the offset to create a new "global" dof handler.
    # Warning : it's not just "add" the offset (mapping += offset), because we desire a dense global
    # numbering, and we don't know where the ghost nodes are
    iglob = offset + 1
    nd = Bcube.get_ndofs(dhl)
    lid2gid = zeros(Int, nd)
    for idof_l in 1:nd
        if dof2part[idof_l] == my_part
            lid2gid[idof_l] = iglob
            iglob += 1
        end
    end
    #@one_at_a_time (@show loc2glob)

    # Now we need to fetch the values for ghost dofs
    # Option 1 : do every thing we did before with the now-correct `dof2part`
    # Option 2 : iterate until convergence by asking to the original ghost parts
    converged = false
    for _ in 1:nprocs
        @only_root println("new iteration -> iglob") comm
        converged = _update_loc2glob!(
            lid2gid,
            tobesent_part_to_idof_l,
            toberecv_part_to_idof_l,
            comm,
        )
    end
    @assert converged "Iterations to determine global numbering are not converged"

    return lid2gid, dof2part
end

"""
Identify, for each dof on the local mesh, the partition that owns that dof. If a dof is shared by two partitions,
the partitions with the smallest id takes the ownership. The result mapping will actually be accurate only for
dof owned by the local partition. For the other, i.e "ghost dofs", the result will be corrected later.
"""
function _compute_dof2part(
    ghost_tag2part::Dict{Int, Int},
    mesh::Bcube.Mesh,
    dhl::DofHandler,
    my_part::Int,
    nparts::Int,
)
    # Identify three types of dof :
    # - the ones that belongs only to the local mesh
    # - the ones that are shared between the local mesh and another
    # - the ones that are only "ghost dofs"
    # dof2type = zeros(Int, 2) # dof local number -> (inside cell number, ghost number)

    # dof2part = local dof number -> partition responsible for this dof
    # Rq : the init number must be greater than the greatest partition number
    dof2part = ones(Int, Bcube.get_ndofs(dhl)) .* (nparts + 10)

    # dof2tag[:,1] = (local dof index) -> global cell
    # dof2tag[:,2] = (local dof index) -> local index in cell
    #dof2tag = zeros(Int, get_ndofs(dhl), 2)

    ind2tag = Bcube.get_absolute_cell_indices(mesh)
    for icell in 1:ncells(mesh)
        icell_g = ind2tag[icell]
        _isghost = haskey(ghost_tag2part, icell_g)

        # Which partition owns that cell? If cell is ghost, assign ghost partition,
        # otherwise, local partition.
        part = _isghost ? ghost_tag2part[icell_g] : my_part

        # Loop over variables
        idofs = Bcube.get_dof(dhl, icell) # local dofs id of cell icell

        for idof in idofs
            # If the partition that owns this cell has a smaller id that the current
            # partition handling this dof, we replace it
            if part < dof2part[idof]
                dof2part[idof] = part
            end
        end # loop over cell dofs
    end # loop over cells

    return dof2part
end

"""
Identify which proc is sending to the local proc
"""
function _identify_src_partitions(dest_parts::Vector{Int}, comm::MPI.Comm)
    n_dest_parts_loc = length(dest_parts)
    my_part = MPI.Comm_rank(comm) + 1
    nparts = MPI.Comm_size(comm)

    # Version 1 : in two steps
    # First, send number of ghost partitions for each proc
    # Second, send ghost partitions id for each proc
    # -> avoid allocation by MPI

    # Send number of ghost partitions for each proc
    n_dest_parts_glo = MPI.Allgather(n_dest_parts_loc, comm)

    # Second, send ghost partitions id for each proc
    sendrecvbuf = zeros(Int, sum(n_dest_parts_glo))
    offset = sum(n_dest_parts_glo[1:(my_part - 1)])
    sendrecvbuf[(offset + 1):(offset + n_dest_parts_loc)] .= dest_parts
    MPI.Allgatherv!(MPI.VBuffer(sendrecvbuf, n_dest_parts_glo), comm)

    # Filter source partition targeting local partition
    src_parts = Int[]
    sizehint!(src_parts, n_dest_parts_loc) # lucky guess
    for ipart in 1:nparts
        # Skip if local part, irrelevant
        (ipart == my_part) && continue

        # Check if `my_part` is present in the ghost parts of `ipart`
        offset = sum(n_dest_parts_glo[1:(ipart - 1)])
        if my_part ∈ sendrecvbuf[(offset + 1):(offset + n_dest_parts_glo[ipart])]
            push!(src_parts, ipart)
        end
    end

    # Version 2 : in one steps
    # Send, in one time, the number of ghosts partitions followed by id of ghost dofs for each
    #n_dest_parts = length(dest_parts)
    #send_buffer = [n_dest_parts, dest_parts...]

    return src_parts
end

"""
Count the number of dofs that each `src` partition will send to the local partition. This
necessitates in return for the local partition to send this info to `dest` partitions.
"""
function _identify_src_ndofs(
    destdof2part::Dict{Int, Int},
    dest_parts::Vector{Int},
    src_parts::Vector{Int},
    comm::MPI.Comm,
)
    # Get the number of dofs that will be sent to local partition by each src partition
    recv_reqs = MPI.Request[]
    n_src_dofs = [[0] for _ in src_parts] # need a Vector{Vector{Int}} because MPI.Irecv! can't handle an Int but only Vector{Int}
    for (i, ipart) in enumerate(src_parts)
        src = ipart - 1
        push!(recv_reqs, MPI.Irecv!(n_src_dofs[i], comm; source = src))
    end

    # Send the number of ghost dofs to each ghost partition
    toberecv_part2ndofs = Dict{Int, Int}()
    send_reqs = MPI.Request[]
    for ipart in dest_parts
        n_dest_dofs = count(==(ipart), values(destdof2part))
        toberecv_part2ndofs[ipart] = n_dest_dofs
        dest = ipart - 1
        push!(send_reqs, MPI.Isend(n_dest_dofs, comm; dest = dest))
    end

    MPI.Waitall(vcat(recv_reqs, send_reqs))
    #MPI.Waitall!(recv_reqs) # no need to wait for the send_reqs to achieve

    tobesent_part2ndofs =
        Dict(ipart => n_src_dofs[i][1] for (i, ipart) in enumerate(src_parts))
    return tobesent_part2ndofs, toberecv_part2ndofs
end

"""
Build the list of ghost dofs that we are asking to other partitions. Dofs are identified by a variable index in the system,
# a global cell tag, and a loc dof index in the cell for this var
"""
function _build_send_ghost_dofs_identifiers(
    dof2part::Vector{Int},
    ghostdof2part::Dict{Int, Int},
    ghost_parts::Vector{Int},
    ghost_tag2indpart::Dict{Int, NTuple{2, Int}},
    dhl::DofHandler,
)
    # We build a dictionnary whose keys are the `dest` partitions and whoses values are matrices containing
    # the identifier of each ghost dofs. A ghost dof is identified by:
    # - global cell tag
    # - local dof index in cell
    part2dofsid = Dict{Int, Matrix{Int}}(
        ipart => zeros(Int, count(==(ipart), values(ghostdof2part)), _N_INFOS) for
        ipart in ghost_parts
    )

    # Track the original local sys index of the ghost dofs
    send_part_to_idof_l = Dict{Int, Vector{Int}}(
        ipart => zeros(Int, size(part2dofsid[ipart], 1)) for ipart in ghost_parts
    )

    # offset for the entries in the matrices of part2dofsid
    offsets = Dict{Int, Int}(ipart => 1 for ipart in ghost_parts)

    # Loop over ghost cells
    for (ghost_tag, ghost_info) in ghost_tag2indpart
        icell_l, remote_part = ghost_info # local cell index and remote cell partition
        icell_g = ghost_tag # global cell index

        idofs_l = Bcube.get_dof(dhl, icell_l) # all local dofs in this cell

        # Loop over dofs in this cell
        for iloc in 1:get_ndofs(dhl, icell_l)
            idof_l = idofs_l[iloc]
            owner = dof2part[idof_l]

            # if the dof is handled by a remote partition, we need to send the dof to this partition
            # (only if the dof has not already been taken into account)
            if owner == remote_part
                offset = offsets[remote_part]

                # Check if dof has not already been taken care of
                if idof_l in view(send_part_to_idof_l[remote_part], 1:(offset - 1))
                    continue
                end

                part2dofsid[remote_part][offset, :] .= icell_g, iloc
                send_part_to_idof_l[remote_part][offset] = idof_l
                offsets[remote_part] += 1
            end
        end # end loop over dofs
    end # end loop on cells

    return part2dofsid, send_part_to_idof_l
end

"""
Identify local dof index that are asked by `src` partitions

Return, for each `src` partition, the local dof ids asked by the remote partition
"""
function _identify_asked_dofs(
    send_part2dofsid::Dict{Int, Matrix{Int}},
    recv_part2dofsid::Dict{Int, Matrix{Int}},
    dhl::DofHandler,
    tag2ind::Dict{Int, Int},
    comm::MPI.Comm,
)
    # Alias
    src_parts = keys(recv_part2dofsid)

    # Receive asked dofs ids
    send_reqs = MPI.Request[]
    recv_part2dofsid_buff = Dict{Int, Vector{Int}}(
        ipart => vec(transpose(recv_part2dofsid[ipart])) for ipart in src_parts
    )
    for ipart in src_parts
        src = ipart - 1
        buffer = recv_part2dofsid_buff[ipart]
        push!(send_reqs, MPI.Irecv!(buffer, comm; source = src))
    end

    # Send ghost dofs ids
    recv_reqs = MPI.Request[]
    for (ipart, dofsid) in send_part2dofsid
        dest = ipart - 1
        buffer = Array(vec(transpose(dofsid))) # allocates but MPI refuses it otherwise...
        push!(recv_reqs, MPI.Isend(buffer, comm; dest = dest))
    end

    # Wait for all the comm to be over
    MPI.Waitall(vcat(send_reqs, recv_reqs))
    #MPI.Waitall!(recv_reqs)

    # Deal with answer
    recv_part_to_idof_l = Dict{Int, Vector{Int}}(
        ipart => zeros(Int, size(recv_part2dofsid[ipart], 1)) for ipart in src_parts
    )
    for ipart in src_parts
        buff = recv_part2dofsid_buff[ipart]
        buff = transpose(reshape(buff, (_N_INFOS, :)))
        for (irow, info) in enumerate(eachrow(buff))
            icell_g, iloc = info
            icell_l = tag2ind[icell_g]
            idofs_l = Bcube.get_dof(dhl, icell_l)
            recv_part_to_idof_l[ipart][irow] = idofs_l[iloc]
        end
    end

    return recv_part_to_idof_l
end

"""
Update `dof2part` by asking to the ghost partition who is the owner of each ghost dof.

`tobesent_idofs` are the dofs asked by remote partitions : it is an info that the local partition knows and send to others
`toberecv_idofs` are the dofs unknown to the local partition : it is an info asked by the local partition to the others
"""
function _update_dof2part!(
    dof2part::Vector{Int},
    tobesent_idofs::Dict{Int, Vector{Int}},
    toberecv_idofs::Dict{Int, Vector{Int}},
    comm::MPI.Comm,
)
    # Async recv
    recv_reqs = MPI.Request[]
    recv_dof2part = Dict{Int, Vector{Int}}(
        ipart => zeros(Int, size(toberecv_idofs[ipart])) for ipart in keys(toberecv_idofs)
    )
    for ipart in keys(toberecv_idofs)
        src = ipart - 1
        buffer = recv_dof2part[ipart]
        push!(recv_reqs, MPI.Irecv!(buffer, comm; source = src))
    end

    # Async send
    send_reqs = MPI.Request[]
    for (ipart, idofs_l) in tobesent_idofs
        dest = ipart - 1
        buffer = dof2part[idofs_l] # current MPI doesn't support `view(dof2part, idofs_l)``
        push!(send_reqs, MPI.Isend(buffer, comm; dest = dest))
    end

    MPI.Waitall(vcat(send_reqs, recv_reqs))
    #MPI.Waitall!(recv_reqs)

    # Deal with answers
    #converged = true # for an unknown reason, using Bool fails on the supercomputer
    _converged = 1 # ... so we use integers instead
    for (ipart, idofs_l) in toberecv_idofs
        _recv_dof2part = recv_dof2part[ipart]
        for (i, idof_l) in enumerate(idofs_l)
            idof_l = idofs_l[i]
            if dof2part[idof_l] > _recv_dof2part[i]
                dof2part[idof_l] = _recv_dof2part[i]
                #converged = false
                _converged = 0
            end
        end
    end

    # Need to obtain the status of all procs
    #MPI.Allreduce!(keep_going, MPI.LOR, comm)
    #converged = MPI.Allreduce(converged, MPI.LOR, comm)
    converged = Bool(MPI.Allreduce(_converged, MPI.LAND, comm))
    # converged = Bool(MPI.Allreduce(_converged, MPI.LOR, comm)) # Why is it a LOR and not a LAND?

    return converged
end

"""
Update `loc2glob`.
`send_idofs_l` is a dict (ipart => idofs_l)
"""
function _update_loc2glob!(
    loc2glob::Vector{Int},
    tobesent_idofs_l::Dict{Int, Vector{Int}},
    toberecv_idofs_l::Dict{Int, Vector{Int}},
    comm::MPI.Comm,
)
    # Async recv
    recv_reqs = MPI.Request[]
    recv_idofs_g = Dict{Int, Vector{Int}}(
        ipart => zeros(Int, size(toberecv_idofs_l[ipart])) for
        ipart in keys(toberecv_idofs_l)
    )
    for ipart in keys(toberecv_idofs_l)
        src = ipart - 1
        buffer = recv_idofs_g[ipart]
        push!(recv_reqs, MPI.Irecv!(buffer, comm; source = src))
    end

    # Async send
    send_reqs = MPI.Request[]
    for (ipart, idofs_l) in tobesent_idofs_l
        dest = ipart - 1
        buffer = loc2glob[idofs_l] # current MPI doesn't support `view(dof2part, idofs_l)`
        push!(send_reqs, MPI.Isend(buffer, comm; dest = dest))
    end

    MPI.Waitall(vcat(send_reqs, recv_reqs))
    #MPI.Waitall!(recv_reqs)

    # Deal with answers
    #converged = true # for an unknown reason, using Bool fails on the supercomputer
    _converged = 1
    for (ipart, idofs_l) in toberecv_idofs_l
        _recv_idofs_g = recv_idofs_g[ipart]
        for (i, idof_l) in enumerate(idofs_l)
            idof_l = idofs_l[i]
            # Check that the received dof is relevant AND this dof is not already known
            # (the second part is important for convergence)
            if ((_recv_idofs_g[i] > 0) && (loc2glob[idof_l] == 0))
                loc2glob[idof_l] = _recv_idofs_g[i]
                #converged = false
                _converged = 0
            end
        end
    end

    # Need to obtain the status of all procs
    #MPI.Allreduce!(keep_going, MPI.LOR, comm)
    #converged = MPI.Allreduce(converged, MPI.LAND, comm)
    converged = Bool(MPI.Allreduce(_converged, MPI.LAND, comm))

    return converged
end

""" debug function """
function printcellcenters(mesh)
    c2n = connectivities_indices(mesh, :c2n)
    for icell in 1:ncells(mesh)
        n = get_nodes(mesh, c2n[icell])
        c = Bcube.center(n, cells(mesh)[icell])
        println("$icell <--> ($(c[1]), $(c[2])")
    end
end
