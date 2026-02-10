# run with `mpirun -n 2 julia flat_heater.jl -ksp_type preonly -pc_type lu`
module flat_heater #hide
println("Running flat heater API example...") #hide

const dir = string(@__DIR__, "/../") # BcubeMPI dir
using MPI
using PetscWrap
using BcubeMPI
using Bcube
using LinearAlgebra
using WriteVTK
using Printf
using SparseArrays
include(joinpath(@__DIR__, "petsc_utils.jl"))

# Include GMSH lib
const key = "GMSH_DIR"
if haskey(ENV, key)
    path = joinpath(ENV[key], "lib/gmsh.jl")
    if isfile(path)
        include(path)
    else
        error("File not found : '$path', please check your '$key' env variable.")
    end
else
    error("You must define a '$key' env variable indicating the path to the gmsh folder.")
end
import Gmsh: gmsh

function gen_mesh(path, npartitions; verbose = false)
    lc = 0.006
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", Int(verbose))
    gmsh.option.setNumber("Mesh.PartitionSplitMeshFiles", 1)
    gmsh.option.setNumber("Mesh.PartitionCreateGhostCells", 1)

    geo = gmsh.model.geo
    P1 = geo.addPoint(0, 0, 0, lc)
    P2 = geo.addPoint(0.5, 0, 0, lc)
    P3 = geo.addPoint(0.5, 0.1, 0, lc)
    P4 = geo.addPoint(0.3, 0.1, 0, lc)
    P5 = geo.addPoint(0.3, 0.09, 0, lc)
    P6 = geo.addPoint(0.17, 0.09, 0, lc)
    P7 = geo.addPoint(0.17, 0.1, 0, lc)
    P8 = geo.addPoint(0.3, 0.11, 0, lc)
    P9 = geo.addPoint(0.17, 0.11, 0, lc)
    P10 = geo.addPoint(0.5, 0.17, 0, lc)
    P11 = geo.addPoint(0, 0.17, 0, lc)
    P12 = geo.addPoint(0, 0.1, 0, lc)

    L1 = geo.addLine(P1, P2)
    L2 = geo.addLine(P2, P3)
    L3 = geo.addLine(P3, P4)
    L4 = geo.addLine(P4, P5)
    L5 = geo.addLine(P5, P6)
    L6 = geo.addLine(P6, P7)
    L7 = geo.addLine(P7, P12)
    L8 = geo.addLine(P12, P1)
    L9 = geo.addLine(P7, P9)
    L10 = geo.addLine(P9, P8)
    L11 = geo.addLine(P8, P4)
    L12 = geo.addLine(P12, P11)
    L13 = geo.addLine(P11, P10)
    L14 = geo.addLine(P10, P3)

    LL1 = geo.addCurveLoop([L1, L2, L3, L4, L5, L6, L7, L8])
    S1 = geo.addPlaneSurface([LL1])
    LL2 = geo.addCurveLoop([L5, L6, L9, L10, L11, L4])
    S2 = geo.addPlaneSurface([LL2])
    LL3 = geo.addCurveLoop([L3, -L11, -L10, -L9, L7, L12, L13, L14])
    S3 = geo.addPlaneSurface([LL3])

    geo.synchronize()

    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [L13]), "FRONT")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [L1]), "REAR")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [L2, L14]), "LEFT")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [L8, L12]), "RIGHT")
    gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [S1]), "MAT_1")
    gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [S3]), "MAT_2")
    gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [S2]), "HEATER")

    mesh = gmsh.model.mesh
    mesh.generate(2)
    mesh.partition(npartitions)

    gmsh.write(path)
    gmsh.finalize()
end

function f1(u, v, params)
    λ, = v
    return params.η * ∇(λ) * transpose(∇(λ))
end

function f2(u, v, params)
    λ, = v
    return params.ρCp * λ * transpose(λ)
end

function f3(u, v, params)
    λ, = v
    return params.q * λ
end

function f4(u, v, params)
    λ, = v
    (params.htc * λ * transpose(λ),)
end

function f5(u, v, params)
    λ, = v
    (params.htc * params.Tr * λ,)
end

# convective boundary condition
const htc = 10000.0
const Tr  = 260.0
const phi = 0.0

# heat source
const l1_h = 60.0e-3
const l2_h = 300.0e-3
const e_h = 0.2e-3

const qtot = 50.0
const qheat = qtot / (l1_h * l2_h * e_h)

const degree = 1

function run()
    # MPI and PETSc init
    # MPI.Init() # PetscInitialize does call "MPI.Init"
    PetscInitialize() # should use  `comm` but not yet available in PetscWrap
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    me = rank + 1

    # Gen mesh
    mesh_path = joinpath(dir, "myout/tmp")
    @only_root gen_mesh(mesh_path * ".msh", nprocs)
    MPI.Barrier(comm)

    # Read mesh
    mesh_path *= "_$me.msh"
    _mesh, el_names, el_names_inv, el_cells, glo2loc_cell_indices =
        read_msh_with_cell_names(mesh_path, 2)
    ghost_tag2part = read_ghosts(mesh_path)
    dmesh = DistributedMesh(_mesh, ghost_tag2part)
    mesh = dmesh.mesh # alias

    fs = FunctionSpace(:Lagrange, degree)
    fes = FESpace(fs, :continuous; size = 1) #  size=1 for scalar variable
    ϕ = CellVariable(:ϕ, dmesh, fes)
    dsys = DistributedSystem(ϕ, dmesh)

    # Create a `TestFunction`
    λ = TestFunction(mesh, fes)

    u, v = ((ϕ,), (λ,))

    # Define measures for cell and interior face integrations
    dΩ = Measure(CellDomain(dmesh), 2 * degree + 1)

    nd = ndofs(ϕ)

    #Adense = zeros(Float64, (nd,nd))
    #Mdense = zeros(Float64, (nd,nd))
    L = zeros(Float64, (nd))

    qtmp = zeros(Float64, (ncells(mesh)))
    heater = vcat(el_cells[el_names_inv["HEATER"]])
    for i in heater
        qtmp[glo2loc_cell_indices[i]] = qheat
    end
    volTag = zeros(Int64, (ncells(mesh)))
    for k in el_names
        elements = el_cells[el_names_inv[k[2]]]
        for i in elements
            volTag[glo2loc_cell_indices[i]] = k[1]
        end
    end

    mat_1 = el_cells[el_names_inv["MAT_1"]]
    mat_2 = el_cells[el_names_inv["MAT_2"]]

    rho = zeros(Float64, (ncells(mesh)))
    cp = zeros(Float64, (ncells(mesh)))
    lamda = zeros(Float64, (ncells(mesh)))
    rhoCp = zeros(Float64, (ncells(mesh)))
    for i in heater
        rho[glo2loc_cell_indices[i]] = 1500.0
        cp[glo2loc_cell_indices[i]] = 900.0
        lamda[glo2loc_cell_indices[i]] = 120.0
        rhoCp[glo2loc_cell_indices[i]] =
            rho[glo2loc_cell_indices[i]] * cp[glo2loc_cell_indices[i]]
    end
    for i in mat_1
        rho[glo2loc_cell_indices[i]] = 2000.0
        cp[glo2loc_cell_indices[i]] = 1000.0
        lamda[glo2loc_cell_indices[i]] = 120.0
        rhoCp[glo2loc_cell_indices[i]] =
            rho[glo2loc_cell_indices[i]] * cp[glo2loc_cell_indices[i]]
    end
    for i in mat_2
        rho[glo2loc_cell_indices[i]] = 2500.0
        cp[glo2loc_cell_indices[i]] = 900.0
        lamda[glo2loc_cell_indices[i]] = 10.0
        rhoCp[glo2loc_cell_indices[i]] =
            rho[glo2loc_cell_indices[i]] * cp[glo2loc_cell_indices[i]]
    end

    #set_values!(q, qtmp)
    #set_values!(ρ, rho)
    #set_values!(Cp, cp)
    #set_values!(η, lamda)

    q = CellData(qtmp)
    ρCp = CellData(rhoCp)
    η = CellData(lamda)

    params = (q = q, ρCp = ρCp, η = η, htc = htc, Tr = Tr)
    #params = (q = 0.0, ρCp= 1.0e6, η=160.0, htc=300.0, Tr=280.0)

    ndm   = max_ndofs(ϕ)
    nhint = ndm * ndm * ncells(mesh)
    Aval  = Float64[]
    rowA  = Int[]
    colA  = Int[]
    sizehint!(Aval, nhint)
    sizehint!(rowA, nhint)
    sizehint!(colA, nhint)

    dict_vars = Dict(
        "rhoCp" => (get_values(ρCp), VTKCellData()),
        "lam" => (get_values(η), VTKCellData()),
        "qheat" => (get_values(q), VTKCellData()),
    )
    write_vtk(dir * "myout/params_flat_heater_$rank", 0, 0.0, mesh, dict_vars)

    Γ_front  = BoundaryFaceDomain(mesh, ("FRONT",))
    dΓ_front = Measure(Γ_front, 2 * degree + 1)

    _AFR = ∫(f4(u, v, params))dΓ_front
    _LFR = ∫(f5(u, v, params))dΓ_front

    # compute matrices associated to bilinear and linear forms
    _A = ∫(f1(u, v, params))dΩ
    _M = ∫(f2(u, v, params))dΩ
    _L = ∫(f3(u, v, params))dΩ

    for (ic, val) in result(_L)
        idof = get_dof(ϕ, ic)
        ndof = length(idof)

        for i in 1:ndof
            L[idof[i]] += val[i]
        end
    end

    for FR_res in _AFR.result
        ic = FR_res[1][1]

        idof = get_dof(ϕ, ic)
        ndof = length(idof)

        #@show ic, idof
        for i in 1:length(idof)
            for j in 1:length(idof)
                push!(Aval, FR_res[3][1][i, j])
                push!(rowA, idof[i])
                push!(colA, idof[j])
            end
        end
    end

    for LFR_res in _LFR.result
        ic = LFR_res[1][1]

        idof = get_dof(ϕ, ic)
        ndof = length(idof)

        for i in 1:length(idof)
            L[idof[i]] += LFR_res[3][1][i]
        end
    end

    AFR = sparse(rowA, colA, Aval, nd, nd)
    #M = sparse(rowM,colM,Mval, nd, nd)
    #A = Adense
    #M = Mdense

    A = sparse(_A, ϕ) + AFR
    M = sparse(_M, ϕ)

    time = 0.0
    dt = 0.1
    totalTime = 10.0

    Miter = (M + dt * A)

    #U0 = 260.0*ones(Float64, nd)
    U1 = 260.0 * ones(Float64, nd)

    # set_values!(ϕ, x -> 260.0) Not implemented for continuous elements
    Bcube.set_values!(ϕ, U1)
    # here a Dirichlet boundary condition is applied on "West". p and T are imposed to 1 (solid).
    #for idof in bnd_dofs["REAR"]
    #    Miter[idof,:] .= 0.0
    #    Miter[idof,idof]  = 1.0
    #    U0[idof] = 300.0
    #end

    # Create corresponding PetscMat
    println("passage")
    Miter_p = julia_sparse_to_petsc(Miter, dsys)
    M_p = julia_sparse_to_petsc(M, dsys)

    # PETSc vectors
    L_p = create_vector(PETSC_DECIDE, ndofs_loc(dsys))
    set_from_options!(L_p)
    update_petscvec!(L_p, L, dsys.localdofs)

    RHS_p, U1_p = duplicate(L_p, 2)
    update_petscvec!(U1_p, U1, dsys.localdofs)

    # Fill vec (useless here...)
    #update_petscvec!(RHS_p, RHS, dsys.localdofs)

    # Set up the linear solver
    ksp = create_ksp(Miter_p)
    set_from_options!(ksp)
    set_up!(ksp)

    #dict_vars = Dict(@sprintf("Temperature") => (get_values(ϕ), VTKPointData()))
    dict_vars = Dict("Temperature" => (var_on_centers(ϕ), VTKCellData()))
    write_vtk(dir * "myout/result_flat_heater_$rank", 0, 0.0, mesh, dict_vars)

    itime = 0
    while time <= totalTime
        time = time + dt
        itime = itime + 1
        @only_root (@show time, itime)
        MatMultAdd(M_p, U1_p, dt * L_p, RHS_p) # corresponds to `RHS_p = dt*L_p + M_p*U1_p`
        #RHS = dt*L + M*U1
        #update_petscvec!(RHS_p, RHS, dsys.localdofs)

        # Solve the system
        solve!(ksp, RHS_p, U1_p)

        # Copy solution to julia vec
        array, _ = VecGetArray(U1_p) # TODO: need to see if this call leads to allocations or not
        U1[dsys.localdofs] .= array

        #U1 .= Miter\RHS # debug

        #for idof in bnd_dofs["REAR"]
        #    RHS[idof] = 300.0
        #end
        #U1 .= Miter\RHS
        #U0 .= U1

        Bcube.set_values!(ϕ, U1)
        update_ghost_dofs!(dsys)

        if itime % 10 == 0
            #dict_vars = Dict(@sprintf("Temperature") => (get_values(ϕ), VTKPointData()))
            dict_vars = Dict("Temperature" => (var_on_centers(ϕ), VTKCellData()))
            write_vtk(dir * "myout/result_flat_heater_$rank", itime, time, mesh, dict_vars)
        end
    end

    # Free
    destroy!(ksp)
    PetscFinalize()
    MPI.Finalize()
end

run()

end #hide
