mutable struct VtkHandler
    basename
    ite
    mesh
    VtkHandler(basename, mesh) = new(basename, 0, mesh)
end

function upwind(ui, uj, nij)
    cij = c ⋅ nij
    if cij > zero(cij)
        flux = cij * ui
    else
        flux = cij * uj
    end
    flux
end
