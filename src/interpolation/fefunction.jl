"""
dev notes : I think this `DistributedSingleFieldFEFunction` is useless, we could only
specialize `set_dof_values` etc for DistributedSingleFESpace
"""
struct DistributedSingleFieldFEFunction{S, FE <: Bcube.AbstractFESpace, V} <:
       Bcube.AbstractSingleFieldFEFunction{S}
    feFunction::Bcube.SingleFieldFEFunction{S, FE, V}

    function DistributedSingleFieldFEFunction(
        f::Bcube.SingleFieldFEFunction{S, FE, V},
    ) where {S, FE, V}
        new{S, FE, V}(f)
    end
end

const DSingleFieldFEFunction = DistributedSingleFieldFEFunction

Base.parent(dfeFunction::DSingleFieldFEFunction) = dfeFunction.feFunction

function Bcube.FEFunction(
    feSpace::Bcube.TrialOrTest{S, FE},
    dofValues = Bcube.allocate_dofs(feSpace);
    updateGhosts::Bool = true,
) where {S, FE <: AbstractDistributedSingleFESpace}
    updateGhosts && HauntedArrays.update_ghosts!(dofValues)
    feFunction = SingleFieldFEFunction(feSpace, dofValues)
    return DistributedSingleFieldFEFunction(feFunction)
end

function Bcube.get_dof_values(f::DSingleFieldFEFunction; updateGhosts::Bool = false)
    updateGhosts && update_ghosts!(f)
    return get_dof_values(parent(f))
end

function Bcube.get_dof_values(f::DSingleFieldFEFunction, icell, n::Val{N}) where {N}
    return Bcube.get_dof_values(parent(f), icell, n)
end

function Bcube.set_dof_values!(
    f::DSingleFieldFEFunction,
    values::AbstractArray;
    updateGhosts = true,
)
    set_dof_values!(parent(f), values)
    updateGhosts && update_ghosts!(f)
end

# Wrap unary function(s)
for op in (:get_fespace,)
    eval(quote
        @inline function Bcube.$op(dfeSpace::DSingleFieldFEFunction)
            Bcube.$op(parent(dfeSpace))
        end
    end)
end

function HauntedArrays.update_ghosts!(f::DSingleFieldFEFunction)
    HauntedArrays.update_ghosts!(get_dof_values(parent(f)))
end

function Base.getproperty(f::DSingleFieldFEFunction, s::Symbol)
    if s === :dofValues
        return getproperty(parent(f), s)
    else # fallback to getfield
        return getfield(f, s)
    end
end
