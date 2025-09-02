# Note : I am not very satisfied because AbstractDistributedSingleFESpace is not a AbstractDistributedFESpace...
# But remember the main constraint : TrialFESpace{FE} where FE must <: AbstractSingleFESpace...
abstract type AbstractDistributedFESpace{S} <: AbstractFESpace{S} end
abstract type AbstractDistributedSingleFESpace{S, FS} <: AbstractSingleFESpace{S, FS} end

struct DistributedSingleFESpace{
    S,
    FS <: AbstractFunctionSpace,
    E <: HauntedArrays.AbstractExchanger,
    I <: Integer,
    C,
} <: AbstractDistributedSingleFESpace{S, FS}
    # Parent SingleFESpace (decorator parent)
    feSpace::SingleFESpace{S, FS}

    # Exchanger (parallel)
    exchanger::E

    # local to global numbering
    lid2gid::Vector{I}

    # Local index to partition owning the dof (partitions start at 1)
    lid2part::Vector{Int}

    # Owned index to local index
    oid2lid::Vector{Int}

    # Cache type for HauntedArrays (storing it as type parameter
    # should be enough, but I prefer to keep it that way for now for clarity)
    # cacheType::C

    function DistributedSingleFESpace(
        f::SingleFESpace{S, FS},
        e,
        l2g,
        l2p,
        o2l,
        C::Type{<:HauntedArrays.AbstractCache},
    ) where {S, FS}
        new{S, FS, typeof(e), eltype(l2g), C}(f, e, l2g, l2p, o2l)
    end
end

const DSingleFESpace = DistributedSingleFESpace

Base.parent(dfeSpace::DSingleFESpace) = dfeSpace.feSpace

@inline get_exchanger(dfeSpace::DSingleFESpace) = dfeSpace.exchanger
@inline get_comm(dfeSpace::DSingleFESpace) = HauntedArrays.get_comm(get_exchanger(dfeSpace))
@inline local_to_part(dfeSpace::DSingleFESpace) = dfeSpace.lid2part
@inline local_to_global(dfeSpace::DSingleFESpace) = dfeSpace.lid2gid
@inline own_to_local(dfeSpace::DSingleFESpace) = dfeSpace.oid2lid
@inline _get_cache_type(::DSingleFESpace{S, FS, E, I, C}) where {S, FS, E, I, C} = C
# @inline _get_cache_type(dfeSpace::DSingleFESpace) = dfeSpace.cacheType

"""
Build the parallel exchanger for a DSingleFESpace. More precisely, it computes the dofs local to global
numbering.
"""
function _build_single_exchanger(dhl::DofHandler, dmesh::DMesh)
    # Build numbering
    lid2gid, lid2part = compute_dof_global_numbering(dhl, dmesh)

    # Build the exchanger
    comm = get_comm(dmesh)

    # Build exchanger
    exchanger = HauntedArrays.MPIExchanger(comm, lid2gid, lid2part)

    # Build additionnal necessary infos
    mypart = MPI.Comm_rank(comm) + 1
    oids = findall(part -> part == mypart, lid2part)
    ghids = findall(part -> part != mypart, lid2part)

    return (exchanger, lid2gid, lid2part, oids, ghids)
end

function Bcube.SingleFESpace(
    fSpace::AbstractFunctionSpace,
    dmesh::DMesh,
    dirichletBndNames = String[];
    size::Int = 1,
    isContinuous::Bool = true,
    kwargs...,
)
    # Retrieve cache type (if provided)
    cacheType = haskey(kwargs, :cacheType) ? kwargs[:cacheType] : HauntedArrays.EmptyCache

    # Build parent FESpace
    feSpace =
        Bcube.SingleFESpace(fSpace, parent(dmesh), dirichletBndNames; size, isContinuous)

    # Build parallel exchanger
    exchanger, lid2gid, lid2part, oids, ghids =
        _build_single_exchanger(Bcube._get_dhl(feSpace), dmesh)

    return DistributedSingleFESpace(feSpace, exchanger, lid2gid, lid2part, oids, cacheType)
end

function Bcube.allocate_dofs(dfeSpace::DSingleFESpace, T = Float64)
    return HauntedArray(
        get_exchanger(dfeSpace),
        local_to_global(dfeSpace),
        local_to_part(dfeSpace),
        own_to_local(dfeSpace),
        1,
        T,
        _get_cache_type(dfeSpace),
    )
end

"""
In `Bcube`, i.e on the local processor, there are two numberings:
* one "local" numbering for each SingleFESpace
* one "global" numbering for the MultiFESpace
These two numberings exist wether the computation is sequential or parallel

In `BcubeParallel` there are the two previous numberings, specific to each rank;
and there are two additionnal numberings:
* one "global" (MPI sense) numbering for each DSingleFESpace
* one "global" (MPI sense) numbering for the DMultiFESpace

For the latter, we have the choice between
* storing one big vector Bcube.MultiFESpace.global --> BcubeParallel.DMultiFESpace.global
* storing, similary to MultiFESpace, a Tuple of Bcube.SingleFESpace.local -> BcubeParallel.DMultiFESpace.global

For now, I keep the two solutions. Names may evolve
"""
struct DistributedMultiFESpace{N, FE, I, E} <: Bcube.AbstractMultiFESpace{N, FE}
    mfeSpace::MultiFESpace{N, FE}

    # Bcube.MultiFESpace.global index to BcubeParallel.DMultiFESpace.global index
    # "sequential" global to "parallel" global
    seqg_to_parg::Vector{I}

    # Bcube.SingleFESpace.local index to BcubeParallel.DMultiFESpace.global index
    seql_to_parg::NTuple{N, Vector{I}}

    # Below : stuff more specific to HauntedArrays, should never be accessed by user

    # Sequential global to partition
    lid2part::Vector{Int}

    # Owned sequential global to sequential global
    oid2lid::Vector{I}

    # Parallel exchanger (to build HauntedArrays)
    exchanger::E

    function DistributedMultiFESpace(
        mfeSpace::MultiFESpace{N, FE},
        seqg_to_parg::Vector{I},
        seql_to_parg::NTuple{N, Vector{I}},
        l2p::Vector{Int},
        o2l::Vector{I},
        exchanger::HauntedArrays.AbstractExchanger,
    ) where {N, I <: Int, FE}
        new{N, FE, I, typeof(exchanger)}(
            mfeSpace,
            seqg_to_parg,
            seql_to_parg,
            l2p,
            o2l,
            exchanger,
        )
    end
end

const DMultiFESpace = DistributedMultiFESpace
Base.parent(dmfeSpace::DMultiFESpace) = dmfeSpace.mfeSpace
@inline get_exchanger(dmfeSpace::DMultiFESpace) = dmfeSpace.exchanger
@inline local_to_global(dmfeSpace::DMultiFESpace) = dmfeSpace.seqg_to_parg
@inline local_to_part(dmfeSpace::DMultiFESpace) = dmfeSpace.lid2part
@inline own_to_local(dmfeSpace::DMultiFESpace) = dmfeSpace.oid2lid

function Bcube.MultiFESpace(
    feSpace::Bcube.TrialOrTest{S, FE},
    feSpaces::Vararg{Bcube.TrialOrTest};
    arrayOfStruct::Bool = Bcube.AOS_DEFAULT,
) where {S, FE <: AbstractDistributedSingleFESpace}
    # Note on the signature function : trick to
    # dispatch here when AbstractDistributedSingleFESpace are used

    # Check that all spaces are <: AbstractDistributedSingleFESpace
    @assert all(_feSpace -> parent(_feSpace) isa AbstractDistributedSingleFESpace, feSpaces) "The FESpaces must be distributed"

    # Alias
    comm = get_comm(_distributed_parent(feSpace))

    # Build sequential MultiFESpace (the parent)
    mfeSpace = Bcube._MultiFESpace((feSpace, feSpaces...); arrayOfStruct)

    # Build parallel global numbering
    seqg_to_parg, seql_to_parg = _build_parallel_global_numbering(mfeSpace)

    # Additonnal infos
    lid2part = zeros(Int, length(seqg_to_parg))
    # oid2lid =
    #     zeros(Int, mapreduce(length, +, map(own_to_local ∘ _distributed_parent, mfeSpace)))
    for (iSpace, feSpace) in enumerate(mfeSpace)
        _feSpace = _distributed_parent(feSpace)
        mapping = Bcube.get_mapping(mfeSpace, iSpace)
        lid2part[mapping] .= local_to_part(_feSpace)
        # oid2lid[mapping] .= view(mapping, own_to_local(_feSpace))
    end
    mypart = MPI.Comm_rank(comm) + 1
    oid2lid = findall(part -> part == mypart, lid2part)

    # @one_at_a_time begin
    #     for (iSpace, feSpace) in enumerate(mfeSpace)
    #         println("feSpace $iSpace")
    #         println("--- lid2gid")
    #         display(local_to_global(_dparent(feSpace)))
    #         println("--- lid2part")
    #         display(local_to_part(_dparent(feSpace)))
    #         println("--- mapping")
    #         display(Bcube.get_mapping(mfeSpace, iSpace))
    #     end
    #     println("seql_to_parg")
    #     display(seql_to_parg)
    # end
    # error("dbg")

    # Exchanger
    exchanger = _build_multi_exchanger(comm, seqg_to_parg, lid2part)
    # exchanger = _build_multi_exchanger(mfeSpace) # to be tested

    return DMultiFESpace(mfeSpace, seqg_to_parg, seql_to_parg, lid2part, oid2lid, exchanger)
end

"""
Non-optimized version (involve collective coms)
"""
function _build_multi_exchanger(comm, lid2gid, lid2part)
    return HauntedArrays.MPIExchanger(comm, lid2gid, lid2part)
end

"""
Optimized version:
TODO : test below once the code works and check it still works!
(it may depend on the way we build the global numbering)
"""
function _build_multi_exchanger(mfeSpace::MultiFESpace)
    return HauntedArrays.merge_exchangers(
        map(get_exchanger, mfeSpace),
        Bcube.get_mapping(mfeSpace),
    )
end

function Bcube.allocate_dofs(dmfeSpace::DMultiFESpace, T = Float64)
    return HauntedArray(
        get_exchanger(dmfeSpace),
        local_to_global(dmfeSpace),
        local_to_part(dmfeSpace),
        own_to_local(dmfeSpace),
        1,
        T,
        _get_cache_type(_distributed_parent(get_fespace(dmfeSpace, 1))),
    )
end

"""
There are many ways to proceed (taking into account MultiFESpace.mapping or not for instance)

The different possibilities:
1) stack each 1st FESpace (does not use MultiFESpace.mapping)
01  u_cell1 | proc 1
02  u_cell2 |
03  u_cell3 | proc 2
04  u_cell4 |
05  u_cell5 | proc 3
06  u_cell6 |
07  v_cell1 | proc 1
08  v_cell2 |
09  v_cell3 | proc 2
10  v_cell4 |
11  v_cell5 | proc 3
12  v_cell6 |

2) Stack all FESpace for 1st proc, then following procs (use MultiFESpace.mapping -> SoA)
01  u_cell1 | proc 1
02  u_cell2 |
03  v_cell1 | proc 1
04  v_cell2 |
05  u_cell3 | proc 2
06  u_cell4 |
07  v_cell3 | proc 2
08  v_cell4 |
09  u_cell5 | proc 3
10  u_cell6 |
11  v_cell5 | proc 3
12  v_cell6 |

3) interlace all FESpace on 1st proc, then following procs (use MultiFESpace.mapping -> AoS)
01  u_cell1 | proc 1
02  v_cell1 |
03  u_cell2 | proc 1
04  v_cell2 |
05  u_cell3 | proc 2
06  v_cell4 |
07  u_cell3 | proc 2
08  v_cell4 |
09  u_cell5 | proc 3
10  v_cell6 |
11  u_cell5 | proc 3
12  v_cell6 |
"""
function _build_parallel_global_numbering(mfeSpace::MultiFESpace)
    _build_parallel_global_numbering_23(mfeSpace)
end

function _build_parallel_global_numbering_1(mfeSpace::MultiFESpace)
    # Alias
    nFESpaces = length(mfeSpace)

    feSpace = get_fespace(mfeSpace, 1) # TrialOrTest
    feSpace = parent(feSpace) # DSingleFE
    comm = get_comm(feSpace)

    # mypart = MPI.Comm_rank(comm) + 1
    np = MPI.Comm_size(comm)

    # Number of "owned" dofs by rank
    ndofs_by_space = [length(own_to_local(feSpace)) for feSpace in mfeSpace]
    ndofs_by_space_and_rank = MPI.Allgather(ndofs_by_space, comm)

    # Reshape : n-feSpace x n-procs
    ndofs = reshape(ndofs_by_space_and_rank, (nFESpaces, :))

    seql_to_parg = ntuple(i -> zero(local_to_global(get_fespace(mfeSpace, i))), nFESpaces)
    seqg_to_parg =
        zeros(eltype(local_to_global(feSpace)), mapreduce(length, +, seql_to_parg))
    for (iSpace, feSpace) in enumerate(mfeSpace)
        # Alias
        seq_mapping = get_mapping(mfeSpace, iSpace)

        # Compute offset by rank for this FESpace
        # For each FESpace, we sum all ndofs from "previous" FESpace on all
        # the "previous" FESpace.
        offset_by_part = ntuple(ipart -> sum(ndofs[1:iSpace, 1:(ipart - 1)]), np)

        for (li, gi, part) in
            enumerate(zip(local_to_global(feSpace), local_to_part(feSpace)))
            seql_to_parg[iSpace][li] = gi + offset_by_part[part]

            seqg_to_parg[seq_mapping[li]] = seql_to_parg[iSpace][li]
        end
    end
    error("this method has never been validated!")
    return seqg_to_parg, seql_to_parg
end

function _build_parallel_global_numbering_23(mfeSpace::MultiFESpace)
    feSpace1 = _distributed_parent(get_fespace(mfeSpace, 1))
    comm = get_comm(feSpace1)

    mypart = MPI.Comm_rank(comm) + 1

    # Total number of "owned" dofs by rank
    ndofs = sum(feSpace -> length(own_to_local(_dparent(feSpace))), mfeSpace)
    ndofs_rank = MPI.Allgather(ndofs, comm)

    # Apply offset on "mappings" -> only valid for own dof. It is corrected later
    offset = sum(ndofs_rank[1:(mypart - 1)])
    seql_to_parg = map(enumerate(mfeSpace)) do (iSpace, feSpace)
        _feSpace = _dparent(feSpace)
        # array = Bcube.allocate_dofs(_feSpace, eltype(local_to_global(_feSpace))) # -> lead to deadlock when PetscCache, don't know why
        array = HauntedArray(
            get_exchanger(_feSpace),
            local_to_global(_feSpace),
            local_to_part(_feSpace),
            own_to_local(_feSpace),
            1,
            eltype(local_to_global(_feSpace)),
        )
        array .= Bcube.get_mapping(mfeSpace, iSpace) .+ offset
        HauntedArrays.update_ghosts!(array)
        parent(array)
    end
    seql_to_parg = (seql_to_parg...,) # turn array into tuple

    # Build seqg_to_parg
    seqg_to_parg =
        zeros(eltype(local_to_global(feSpace1)), mapreduce(length, +, seql_to_parg))
    for iSpace in 1:length(mfeSpace)
        seqg_to_parg[Bcube.get_mapping(mfeSpace, iSpace)] .= seql_to_parg[iSpace]
    end

    return seqg_to_parg, seql_to_parg
end

"""
AbstractFESpace can be a TrialFESpace or a AbstractSingleFESpace. If it's a Trial,
look for the parent that is a DSingleFESpace. If it's already a DSingleFESpace,
return it.
"""
function _distributed_parent(feSpace::AbstractFESpace, level = 0)
    @assert level <= 1 "Could not find any distributed parent"
    if feSpace isa AbstractDistributedSingleFESpace
        return feSpace
    else
        return _distributed_parent(parent(feSpace), level + 1)
    end
end
const _dparent = _distributed_parent
