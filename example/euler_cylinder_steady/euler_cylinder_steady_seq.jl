module EulerCylinderSteady #hide
println("Running euler_cylinder_steady example...") #hide
# # Solve Euler equation around a cylinder

using Bcube, BcubeGmsh, BcubeVTK
using LinearAlgebra
using StaticArrays
using SparseArrays
using DiffEqCallbacks
using OrdinaryDiffEq
using SparseDiffTools
include(joinpath(@__DIR__, "common_euler.jl"))

mutable struct VtkHandler
    basename::String
    basename_residual::String
    ite::Int
    VtkHandler(basename) = new(basename, basename * "_residual", 0)
end

"""
    Write solution (at cell centers) to vtk
    Wrapper for `write_vtk`
"""
function append_vtk(vtk, mesh, vars, t, params; res = nothing)
    ρ, ρu, ρE = vars

    dict_vars_dg = Dict("ρ" => ρ, "ρu" => ρu, "ρE" => ρE)
    write_file(
        vtk.basename,
        mesh,
        dict_vars_dg,
        vtk.ite,
        t,
        ;
        collection_append = vtk.ite > 0,
    )

    # Update counter
    vtk.ite += 1
end

function main(stateInit, stateBcFarfield, degree)
    @show degree, degquad

    BcubeGmsh.gen_mesh_around_disk(joinpath(outputpath, "mesh.msh"), :tri; nθ = 4 * 10)
    # Bcube.gen_mesh_around_disk(joinpath(outputpath, "mesh.msh"); nθ = 180, nr = 25)
    mesh = read_mesh(joinpath(outputpath, "mesh.msh"))
    # scale!(mesh, 1.0 / 0.5334)

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
    U_sca = TrialFESpace(fs, mesh, :discontinuous; size = 1) # DG, scalar
    U_vec = TrialFESpace(fs, mesh, :discontinuous; size = 2) # DG, vectoriel
    V_sca = TestFESpace(U_sca)
    V_vec = TestFESpace(U_vec)
    U = MultiFESpace(U_sca, U_vec, U_sca)
    V = MultiFESpace(V_sca, V_vec, V_sca)

    q = FEFunction(U)

    # select an initial configurations:
    if deg == 0
        init!(q, mesh, stateInit)
    else
        println("Start projection")
        projection_l2!(q, qLowOrder, dΩ)
        println("End projection")
    end

    # Init vtk handler
    mkpath(outputpath)
    vtk = VtkHandler(
        joinpath(
            outputpath,
            "euler_naca_mdeg" * string(mesh_degree) * "_deg" * string(deg) * ".pvd",
        ),
    )

    # Init time
    time = 0.0

    # Save initial solution
    append_vtk(vtk, mesh, q, time, params)

    # Solve
    # time, q = steady_solve_expl!(U, V, q, mesh, params, vtk)
    time, q = steady_solve_impl_diffeq!(U, V, q, mesh, params, vtk)

    # Save final solution
    append_vtk(vtk, mesh, q, time, params)
    println("end steady_solve for deg=", deg, " !")
end

function steady_solve_impl_diffeq!(U, V, q, mesh, params, vtk)
    counter = [0]

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

    ode_params = (
        U = U,
        V = V,
        l = l,
        params = params,
        counter = counter,
        vtk = vtk,
        cache = (cacheCellMean = Bcube.build_cell_mean_cache(q, dΩ),),
    )

    # compute sparsity pattern and coloring
    println("computing jacobian cache...")
    sparsity_pattern = Bcube.build_jacobian_sparsity_pattern(U, mesh)
    colors = matrix_colors(sparsity_pattern)

    ode = ODEFunction(
        rhs!;
        mass_matrix = Bcube.build_mass_matrix(U, V, params.dΩ),
        jac_prototype = sparsity_pattern,
        colorvec = colors,
    )

    timestepper = ImplicitEuler(; nlsolve = NLNewton(; max_iter = 20))

    Tfinal  = Inf
    problem = ODEProblem(ode, get_dof_values(q), (0.0, Tfinal), ode_params)

    cb_cache  = DiscreteCallback(always_true, update_cache!; save_positions = (false, false))
    cb_vtk    = DiscreteCallback(always_true, output_vtk; save_positions = (false, false))
    cb_steady = TerminateSteadyState(1e-6, 1e-6, condition_steadystate)

    error = 1e-1

    sol = solve(
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
        # isoutofdomain = isoutofdomain,
        callback = CallbackSet(cb_cache, cb_vtk, cb_steady),
    )

    set_dof_values!(q, sol.u[end])
    return sol.t[end], q
end

function isoutofdomain(dof, p, t)
    any(isnan, dof) && return true

    q = FEFunction(p.U, dof)
    q_mean = map(get_values, Bcube.cell_mean(q, p.cache.cacheCellMean))
    p_mean = pressure.(q_mean..., stateInit.γ)

    negative_ρ = any(x -> x < 0, q_mean[1])
    negative_p = any(x -> x < 0, p_mean)
    isout = negative_ρ || negative_p
    isout && @show negative_ρ, negative_p
    return isout
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
    function l(_q, v)
        ∫(flux_Ω(_q, v))dΩ - ∫(flux_Γ(_q, v, nΓ))dΓ -
        ∫(flux_Γ_wall(_q, v, nΓ_wall))dΓ_wall -
        ∫(flux_Γ_farfield(_q, v, nΓ_farfield))dΓ_farfield
    end

    M = Bcube.build_mass_matrix(U, V, params.dΩ)
    Minv = inv(Matrix(M))
    @show get_ndofs(U)

    dt = 1e-6
    t = 0.0
    println("Running explicit iterations...")
    for i in 1:2000
        println("Iteration $i")
        rhs = assemble_linear(v -> l((q...,), v), V)
        dq = dt * Minv * rhs
        # dq = dt * M \ rhs
        set_dof_values!(q, get_dof_values(q) .+ dq)
        t += dt
        # (i % 10 == 0) && append_vtk(vtk, mesh, q, t, params)
    end

    return t, q
end

function update_cache!(integrator)
    U = integrator.p.U
    Q1, = U
    deg = get_degree(Bcube.get_function_space(Q1))
    println(
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
    mesh = get_mesh(get_domain(integrator.p.params.dΩ))
    q = FEFunction(integrator.p.U, integrator.u)
    counter = integrator.p.counter
    counter .+= 1
    if (counter[1] % nout == 0)
        println("output_vtk ", counter[1])
        append_vtk(integrator.p.vtk, mesh, q, integrator.t, integrator.p.params)
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
const outputpath = joinpath(@__DIR__, "..", "..", "tmp", "euler_cylinder_steady_seq")
rm(outputpath; force = true, recursive = true)
mkdir(outputpath)

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
