# Solve Euler equation around a cylinder
# PETCs options:
# -ksp_view
# -pc_type lu
module EulerCylinderSteady #hide
println("Running euler_cylinder_steady example...") #hide
println("loading modules...")
using Bcube, BcubeGmsh
#using StatProfilerHTML # nice to profile in parallel
using BcubeMPI
using LinearAlgebra
using WriteVTK
using StaticArrays
using SparseArrays
using WriteVTK
using OrdinaryDiffEq
using DiffEqCallbacks
using SparseDiffTools
using HauntedArrays
using HauntedArrays2PetscWrap
using PetscWrap
using MPI
using MPIUtils
include(joinpath(@__DIR__, "common_euler.jl"))
println("done loading modules !")

# Init Petsc, you can pass arguments to control the linear system algorithm
# init_petsc("-pc_type lu") # for a direct solver (MUMPS)
init_petsc() # default iterative solver : ILU(0) + GMRES

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const np = MPI.Comm_size(comm)

mutable struct VtkHandler{M}
    basename::String
    mesh::M
    ite::Int
    VtkHandler(basename, mesh) = new{typeof(mesh)}(basename, mesh, 0)
end

"""
    Write solution (at cell centers) to vtk
    Wrapper for `write_vtk`
"""
function append_vtk(vtk, vars, t, params; res = nothing)
    mesh = parent(vtk.mesh)
    ρ, ρu, ρE = vars
    # println("call to append_vtk")
    # update_ghosts!.(ρ, ρu, ρE) # I don't know why but it's not working (multithreaded broadcast maybe?)
    # update_ghosts!(ρ)
    # update_ghosts!(ρu)
    # update_ghosts!(ρE)

    # Mean cell values
    # name2val_mean = (;zip(get_name.(vars), mean_values.(vars, degquad))...)
    # p_mean = pressure.(name2val_mean[:ρ], name2val_mean[:ρu], name2val_mean[:ρE], params.stateInit.γ)

    # vtk_degree = maximum(x -> get_degree(Bcube.get_function_space(get_fespace(x))), vars)
    # vtk_degree = max(1, mesh_degree, vtk_degree)
    # _ρ = var_on_nodes_discontinuous(ρ, mesh, vtk_degree)
    # _ρu = var_on_nodes_discontinuous(ρu, mesh, vtk_degree)
    # _ρE = var_on_nodes_discontinuous(ρE, mesh, vtk_degree)
    _ρ = var_on_vertices(ρ, mesh)
    _ρu = var_on_vertices(ρu, mesh)
    _ρE = var_on_vertices(ρE, mesh)

    Cp = [pressure_coefficient(x, y, z) for (x, y, z) in zip(_ρ, eachrow(_ρu), _ρE)]
    Ma = [mach(x, y, z) for (x, y, z) in zip(_ρ, eachrow(_ρu), _ρE)]

    dict_vars_dg = Dict(
        "rho" => (_ρ, VTKPointData()),
        "rhou" => (transpose(_ρu), VTKPointData()),
        "rhoE" => (_ρE, VTKPointData()),
        "Cp" => (Cp, VTKPointData()),
        "Mach" => (Ma, VTKPointData()),
        # "rho" => (var_on_centers(ρ, mesh), VTKCellData()),
        # "rho_mean" => (get_values(Bcube.cell_mean(ρ, params.dΩ)), VTKCellData()),
        # "rhou_mean" => (get_values(Bcube.cell_mean(ρu, params.dΩ)), VTKCellData()),
        # "rhoE_mean" => (get_values(Bcube.cell_mean(ρE, params.dΩ)), VTKCellData()),
    )
    # Bcube.write_vtk_discontinuous(
    #     vtk.basename * "_DG",
    #     vtk.ite,
    #     t,
    #     mesh,
    #     dict_vars_dg,
    #     vtk_degree;
    #     append = vtk.ite > 0,
    # )

    BcubeMPI.write_pvtk(
        vtk.basename,
        vtk.ite,
        t,
        vtk.mesh,
        dict_vars_dg;
        append = vtk.ite > 0,
    )

    # Update counter
    vtk.ite += 1

    return nothing
end

function main(stateInit, stateBcFarfield, degree)
    @only_root begin
        println("Building mesh...")
        mkpath(outputpath)
        BcubeGmsh.gen_mesh_around_disk(
            joinpath(outputpath, "mesh.msh"),
            :tri;
            nθ = 4 * 20,
            nr = 25,
            split_files = true,
            n_partitions = np,
            create_ghosts = true,
        )
    end
    MPI.Barrier(comm)
    println("Reading mesh file...")
    mesh = read_partitioned_msh(joinpath(outputpath, "mesh.msh"), comm)
    println("ncells local = $(ncells(parent(mesh)))")

    # Then we create a `NamedTuple` to hold the simulation parameters.
    params = (degquad = degquad, stateInit = stateInit, stateBcFarfield = stateBcFarfield)

    # Define measures for cell and interior face integrations
    dΩ = Measure(CellDomain(mesh), degquad)
    dΓ = Measure(InteriorFaceDomain(mesh), degquad)

    # Declare boundary conditions and
    # create associated domains and measures
    Γ_wall      = BoundaryFaceDomain(mesh, ("Wall",))
    Γ_farfield  = BoundaryFaceDomain(mesh, ("Farfield",))
    dΓ_wall     = Measure(Γ_wall, degquad)
    dΓ_farfield = Measure(Γ_farfield, degquad)

    params = (
        params...,
        Γ_wall = Γ_wall,
        dΓ = dΓ,
        dΩ = dΩ,
        dΓ_wall = dΓ_wall,
        dΓ_farfield = dΓ_farfield,
    )

    deg = degree
    params = (params..., degree = deg)

    fs = FunctionSpace(fspace, deg)
    U_sca = TrialFESpace(fs, mesh, :discontinuous; size = 1, cacheType = PetscCache) # DG, scalar
    U_vec = TrialFESpace(fs, mesh, :discontinuous; size = 2, cacheType = PetscCache) # DG, vectoriel
    V_sca = TestFESpace(U_sca)
    V_vec = TestFESpace(U_vec)
    U = MultiFESpace(U_sca, U_vec, U_sca)
    V = MultiFESpace(V_sca, V_vec, V_sca)
    q = FEFunction(U)

    # select an initial configurations:
    if deg == 0
        init!(q, mesh, stateInit)
    else
        @only_root println("Start projection")
        projection_l2!(q, qLowOrder, dΩ)
        @only_root println("End projection")
    end

    # Init vtk handler
    mkpath(outputpath)
    # vtk = VtkHandler(outputpath * "euler_naca_mdeg$(mesh_degree)_deg$(deg)_r$(rank)")
    vtk = VtkHandler(joinpath(outputpath, "euler_naca_mdeg$(mesh_degree)_deg$(deg)"), mesh)

    # Init time
    time = 0.0

    # Save initial solution
    append_vtk(vtk, q, time, params)

    # Solve
    time, q = steady_solve_expl!(U, V, q, mesh, params, vtk)
    # time, q = steady_solve_impl!(U, V, q, mesh, params, vtk)
    # time, q = steady_solve_impl_diffeq!(U, V, q, mesh, params, vtk)

    # Save final solution
    append_vtk(vtk, q, time, params)
    println("end steady_solve for deg=", deg, " !")
end

function steady_solve_expl!(U, V, q, mesh, params, vtk)
    # alias on measures
    dΓ = params.dΓ
    dΩ = params.dΩ
    dΓ_wall = params.dΓ_wall
    dΓ_farfield = params.dΓ_farfield

    # face normals for each face domain (lazy, no computation at this step)
    nΓ = get_face_normals(dΓ)
    nΓ_wall = get_face_normals(dΓ_wall)
    nΓ_farfield = get_face_normals(dΓ_farfield)

    # Linear form
    function l(q, v)
        ∫(flux_Ω(q, v))dΩ - ∫(flux_Γ(q, v, nΓ))dΓ - ∫(flux_Γ_wall(q, v, nΓ_wall))dΓ_wall -
        ∫(flux_Γ_farfield(q, v, nΓ_farfield))dΓ_farfield
    end

    # Create a HauntedVector initialized with the value of `q`
    # Q0 = get_dof_values(q) # errors since `get_dof_values` is implemented with a `mapreduce`...
    Q = Bcube.allocate_dofs(Bcube._get_mfe_space(q))
    Bcube.get_dof_values!(Q, q)
    @show n_local_rows(Q), n_own_rows(Q)

    # Mass matrix
    @only_root println("building mass matrix...")
    M = Bcube.build_mass_matrix(U, V, dΩ)
    M = HauntedMatrix(M, Q) # sparse matrix -> haunted sparse matrix
    @show all(isfinite.(M.array))

    # Since the mass matrix will be constant, build the PETSC KSP once and for all
    _A = HauntedArrays2PetscWrap.get_updated_petsc_array(M)
    ksp = PetscWrap.create_ksp(_A; autosetup = true, add_finalizer = false)

    p = (U = U, V = V, l = l, params = params, counter = [0], vtk = vtk)

    dt = 1e-6
    t = 0.0
    @only_root println("Running explicit iterations...")
    for i in 1:1000
        @only_root println("Iteration $i")

        HauntedArrays.update_ghosts!(Q)
        rhs = zero(Q)
        rhs!(rhs, Q, p, t)

        # Solve linear system
        _b = HauntedArrays2PetscWrap.get_updated_petsc_array(rhs)
        dQ = similar(rhs)
        _x = HauntedArrays2PetscWrap.get_updated_petsc_array(dQ)
        PetscWrap.solve(ksp, _b, _x)
        HauntedArrays2PetscWrap.update!(dQ, _x)

        Q += dt * dQ
        t += dt
    end

    # Free the KSP
    PetscWrap.destroy(ksp)

    return t, FEFunction(U, Q)
end

function steady_solve_impl!(U, V, q, mesh, params, vtk)
    # alias on measures
    dΓ = params.dΓ
    dΩ = params.dΩ
    dΓ_wall = params.dΓ_wall
    dΓ_farfield = params.dΓ_farfield

    # face normals for each face domain (lazy, no computation at this step)
    nΓ = get_face_normals(dΓ)
    nΓ_wall = get_face_normals(dΓ_wall)
    nΓ_farfield = get_face_normals(dΓ_farfield)

    # Linear form
    function l(q, v)
        ∫(flux_Ω(q, v))dΩ - ∫(flux_Γ(q, v, nΓ))dΓ - ∫(flux_Γ_wall(q, v, nΓ_wall))dΓ_wall -
        ∫(flux_Γ_farfield(q, v, nΓ_farfield))dΓ_farfield
    end

    p = (U = U, V = V, l = l, params = params, counter = [0], vtk = vtk)

    # Create a HauntedVector initialized with the value of `q`
    # Q0 = get_dof_values(q) # errors since `get_dof_values` is implemented with a `mapreduce`...
    Q = Bcube.allocate_dofs(Bcube._get_mfe_space(q))
    Bcube.get_dof_values!(Q, q)
    @show n_local_rows(Q), n_own_rows(Q)

    # Mass matrix
    @only_root println("building mass matrix...")
    M = Bcube.build_mass_matrix(U, V, dΩ)
    M = HauntedMatrix(M, Q) # sparse matrix -> haunted sparse matrix
    @show all(isfinite.(M.array))

    # Compute sparsity pattern and coloring
    @only_root println("computing jacobian cache...")
    sparsity = Bcube.build_jacobian_sparsity_pattern(U, mesh)
    J = HauntedMatrix(Float64.(sparsity) + I, Q) # +I to ensure diag belongs to sparsity pattern
    colorvec = matrix_colors(J)
    f! = (y, x) -> rhs!(y, x, p, 0.0)
    _x = similar(Q)
    _dx = similar(Q)
    jac_cache = ForwardColorJacCache(f!, _x, nothing; dx = _dx, colorvec, sparsity)

    Δt = 1e-6
    t = 0.0
    @only_root println("Running implicit iterations...")
    for i in 1:2000
        @only_root println("Iteration $i")

        HauntedArrays.update_ghosts!(Q)
        rhs = zero(Q)

        # Eval rhs
        rhs!(rhs, Q, p, t)

        # Compute jacobian
        forwarddiff_color_jacobian!(J, f!, Q, jac_cache)

        # Prepare linear system : (M - Δt*J) = Δt * rhs => Ax=b
        parent(J).nzval .*= Δt # necessary to preserve sparsity-pattern, `J .*= Δt` does not!
        A = parent(M) - parent(J)
        A = HauntedMatrix(A, Q)

        # Solve linear system
        _A = HauntedArrays2PetscWrap.get_updated_petsc_array(A)
        _b = HauntedArrays2PetscWrap.get_updated_petsc_array(Δt .* rhs)
        ksp = PetscWrap.create_ksp(_A; autosetup = true, add_finalizer = false)
        dQ = similar(rhs)
        _x = HauntedArrays2PetscWrap.get_updated_petsc_array(dQ)
        PetscWrap.solve(ksp, _b, _x)
        HauntedArrays2PetscWrap.update!(dQ, _x)
        PetscWrap.destroy(ksp)

        Q += dQ
        t += Δt

        if i % 10 == 0
            set_dof_values!(q, Q)
            append_vtk(vtk, q, t, params)
        end
    end

    return t, FEFunction(U, Q)
end

function steady_solve_impl_diffeq!(U, V, q, mesh, params, vtk)
    error("for an unknown reason, this example does work correctly, even on one proc")

    # alias on measures
    dΓ = params.dΓ
    dΩ = params.dΩ
    dΓ_wall = params.dΓ_wall
    dΓ_farfield = params.dΓ_farfield

    # face normals for each face domain (lazy, no computation at this step)
    nΓ = get_face_normals(dΓ)
    nΓ_wall = get_face_normals(dΓ_wall)
    nΓ_farfield = get_face_normals(dΓ_farfield)

    # Linear form
    function l(q, v)
        ∫(flux_Ω(q, v))dΩ - ∫(flux_Γ(q, v, nΓ))dΓ - ∫(flux_Γ_wall(q, v, nΓ_wall))dΓ_wall -
        ∫(flux_Γ_farfield(q, v, nΓ_farfield))dΓ_farfield
    end

    ode_params = (U = U, V = V, l = l, params = params, counter = [0], vtk = vtk)

    # Create a HauntedVector initialized with the value of `q`
    # Q0 = get_dof_values(q) # errors since `get_dof_values` is implemented with a `mapreduce`...
    Q0 = Bcube.allocate_dofs(Bcube._get_mfe_space(q))
    Bcube.get_dof_values!(Q0, q)
    @show n_local_rows(Q0), n_own_rows(Q0)

    # Mass matrix
    @only_root println("building mass matrix...")
    M = Bcube.build_mass_matrix(U, V, dΩ)
    M = HauntedMatrix(M, Q0) # sparse matrix -> haunted sparse matrix
    @show all(isfinite.(M.array))

    # compute sparsity pattern and coloring
    @only_root println("computing jacobian cache...")
    sparsity_pattern = Bcube.build_jacobian_sparsity_pattern(U, mesh)
    jac_prototype = HauntedMatrix(Float64.(sparsity_pattern) + I, Q0) # +I to ensure diag belongs to sparsity pattern
    colorvec = matrix_colors(jac_prototype)

    ode = ODEFunction(rhs!; mass_matrix = M, jac_prototype, colorvec)

    timestepper = ImplicitEuler(;
        linsolve = PetscFactorization(),
        nlsolve = NLNewton(; max_iter = 20),
    )

    Tfinal  = Inf
    problem = ODEProblem(ode, Q0, (0.0, Tfinal), ode_params)

    cb_update = DiscreteCallback(always_true, integrator -> begin
        HauntedArrays.update_ghosts!(integrator.u)
    end; save_positions = (false, false))
    cb_cache  = DiscreteCallback(always_true, update_cache!; save_positions = (false, false))
    cb_vtk    = DiscreteCallback(always_true, output_vtk; save_positions = (false, false))
    cb_steady = TerminateSteadyState(1e-6, 1e-6, condition_steadystate)

    error = 1e-1

    @only_root println("running solve...")
    sol = OrdinaryDiffEq.solve(
        problem,
        timestepper;
        dt = 1e-7,
        initializealg = NoInit(),
        abstol = error,
        reltol = error,
        progress = false,
        progress_steps = 1000,
        save_everystep = false,
        save_start = false,
        save_end = false,
        callback = CallbackSet(cb_update, cb_cache, cb_vtk, cb_steady),
    )

    return sol.t[end], FEFunction(U, sol.u[end])
end

function update_cache!(integrator)
    U = integrator.p.U
    Q1, = U
    deg = get_degree(Bcube.get_function_space(Q1))
    @only_root println(
        "deg=",
        deg,
        " update_cache! : iter=",
        integrator.p.counter[1],
        " dt=",
        integrator.dt,
    )
end

function output_vtk(integrator)
    u_modified!(integrator, false)
    q = FEFunction(integrator.p.U, integrator.u)
    counter = integrator.p.counter
    counter .+= 1
    if (counter[1] % nout == 0)
        @only_root println("output_vtk ", counter[1])
        append_vtk(integrator.p.vtk, q, integrator.t, integrator.p.params)
    end
    return nothing
end

const degree = 0 # Function-space degree
const mesh_degree = 1
const fspace = :Taylor

const stateInit = (
    AoA = deg2rad(0.0),
    M_inf = 0.1,
    P_inf = 101325.0,
    T_inf = 275.0,
    r_gas = 287.0,
    γ = 1.4,
)
const nout = 1 # number of step between two vtk outputs
const degquad = 6
const outputpath = joinpath(@__DIR__, "..", "..", "tmp", "euler_cylinder_steady_par")
@only_root begin
    rm(outputpath; force = true, recursive = true)
    mkdir(outputpath)
end

const stateBcFarfield = (
    AoA = stateInit.AoA,
    M_inf = stateInit.M_inf,
    Pᵢ_inf = compute_Pᵢ(stateInit.P_inf, stateInit.γ, stateInit.M_inf),
    Tᵢ_inf = compute_Tᵢ(stateInit.T_inf, stateInit.γ, stateInit.M_inf),
    u_inf = bc_state_farfield(
        stateInit.AoA,
        stateInit.M_inf,
        stateInit.P_inf,
        stateInit.T_inf,
        stateInit.r_gas,
        stateInit.γ,
    ),
    r_gas = stateInit.r_gas,
    γ = stateInit.γ,
)

main(stateInit, stateBcFarfield, degree)

end #hide
