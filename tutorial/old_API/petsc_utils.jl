function julia_sparse_to_petsc(A::AbstractSparseMatrix, dsys::DistributedSystem)
    # Create corresponding PetscMat
    nd_loc = ndofs_loc(dsys)
    A_petsc = create_matrix(PETSC_DECIDE, PETSC_DECIDE, nd_loc, nd_loc; auto_setup = false) # not sure about column local size
    set_from_options!(A_petsc)

    # Build local sys index -> petsc numbering
    iglob_min, iglob_max = extrema(dsys.loc2glob[dsys.localdofs]) # extrema of global dof index on this partition

    # Retrieve CSR information from SparseArray
    _I, _J, _V = findnz(A)

    # Set exact preallocation)
    d_nnz, o_nnz = preallocation_from_sparse(_I, _J, dsys.loc2glob, iglob_min, iglob_max)
    preallocate!(A_petsc, d_nnz, o_nnz)

    # Fill matrix
    fill_petscmat!(A_petsc, _I, _J, _V, dsys.loc2glob, iglob_min, iglob_max)

    return A_petsc
end

"""
Find number of non-zeros element per diagonal block and off-diagonal block
(see https://petsc.org/release/docs/manualpages/Mat/MatMPIAIJSetPreallocation.html)

`nrows` is the number of rows handled by the local processor. `imin` is the minimum row number
(i.e `min(I)` if all rows contains nnz).
"""
function preallocation_from_sparse(
    I::Vector{Int},
    J::Vector{Int},
    loc2glob::Vector{Int},
    iglob_min::Int,
    iglob_max::Int,
)
    # Allocate
    nrows = iglob_max - iglob_min + 1
    d_nnz = zeros(Int, nrows)
    o_nnz = zeros(Int, nrows)

    # Search for non-zeros
    for (iloc, jloc) in zip(I, J)
        # Check that the row is handled by the local processor
        iglob = loc2glob[iloc]
        (iglob_min <= iglob <= iglob_max) || continue

        # Look if the column belongs to a diagonal block or not
        # Rq: `iglob - iglob_min + 1` and not `iloc` because the ghost
        # dofs are not necessarily at the end, they can be any where in local
        # numbering...
        jglob = loc2glob[jloc]
        if (iglob_min <= jglob <= iglob_max)
            d_nnz[iglob - iglob_min + 1] += 1
        else
            o_nnz[iglob - iglob_min + 1] += 1
        end
    end
    return d_nnz, o_nnz
end

"""
Remark : the global numbering of the dofs handled by the local partition is dense
"""
function fill_petscmat!(
    A::PetscMat,
    I::Vector{PetscInt},
    J::Vector{PetscInt},
    V::Vector{PetscScalar},
    loc2glob::Vector{Int},
    iglob_min::Int,
    iglob_max::Int;
    assemble = true,
)
    offset = get_range(A)[1] - iglob_min
    for (i, j, v) in zip(I, J, V)
        if (iglob_min <= loc2glob[i] <= iglob_max)
            PetscWrap.set_value!(
                A,
                loc2glob[i] + offset,
                loc2glob[j] + offset,
                v,
                PetscWrap.ADD_VALUES,
            )
        end
    end
    assemble && assemble!(A)
end

function fill_petscmat!(
    A::PetscMat,
    I::Vector{T1},
    J::Vector{T1},
    V::Vector{T2},
    loc2glob::Vector{Int},
    iglob_min::Int,
    iglob_max::Int;
    assemble = true,
) where {T1 <: Integer, T2 <: Number}
    fill_petscmat!(
        A,
        PetscInt.(I),
        PetscInt.(J),
        PetscScalar.(V),
        loc2glob,
        iglob_min,
        iglob_max;
        assemble,
    )
end

"""
Remark : the global numbering of the dofs handled by the local partition is dense
"""
function update_petscvec!(b::PetscVec, b_vals::Vector{Float64}, localdofs::Vector{Int})
    PetscWrap.set_values!(b, b_vals[localdofs])
    assemble!(b)
end
