module LinearTransport
println("Running linear transport example...")
println("Loading modules...")
using MPI
using Bcube, BcubeGmsh, BcubeVTK
using BcubeMPI
using LinearAlgebra
using WriteVTK # to be removed when "write_pvtk" will be in BcubeVTK
using MPIUtils
using OrdinaryDiffEq
using HauntedArrays
using Symbolics
using HauntedArrays2PetscWrap
using SparseDiffTools
include(joinpath(@__DIR__, "common.jl"))
println("done loading modules")

function append_vtk(vtk, u::Bcube.AbstractFEFunction, t)
    # Values on centers
    # values = var_on_vertices(u, vtk.mesh)

    # Write
    # values = var_on_nodes_discontinuous(u, vtk.mesh)
    # Bcube.write_vtk_discontinuous(
    #     vtk.basename,
    #     vtk.ite,
    #     t,
    #     vtk.mesh,
    #     Dict("u" => (values, VTKPointData())),
    #     1;
    #     append = vtk.ite > 0,
    # )

    values = var_on_centers(u, parent(vtk.mesh))
    BcubeMPI.write_pvtk(
        vtk.basename,
        vtk.ite,
        t,
        vtk.mesh,
        Dict("u" => (values, VTKCellData()));
        append = vtk.ite > 0,
    )

    # Update counter
    vtk.ite += 1
end

always_true(args...) = true

function timeintegration_expl(m, l, U, V, cbset, dmesh)
    rank = MPI.Comm_rank(BcubeMPI.get_comm(dmesh))
    vtk = VtkHandler(joinpath(out_dir, "linear_transport_expl"), dmesh)

    Δt = CFL * min(lx / nx, ly / ny) / norm(c) # Time step
    M = assemble_bilinear(m, U, V)

    # Parameters and time span to solve
    p = (M = M, U = U, V = V, vtk = vtk, l = l, iter = zeros(Int, 1), time = zeros(1000))
    tspan = (0.0, totalTime)

    prob = ODEProblem(f_expl!, Bcube.allocate_dofs(U), tspan, p)
    @only_root println("Running explicit solve...")
    solve(prob, Euler(); dt = Δt, callback = cbset, save_everystep = false)
end

function timeintegration_impl_dense(m, l, U, V, cbset, dmesh)
    rank = MPI.Comm_rank(BcubeMPI.get_comm(dmesh))
    vtk = VtkHandler(joinpath(out_dir, "linear_transport_impl_dense"), dmesh)

    q0 = Bcube.allocate_dofs(U)

    # Mass matrix
    M = assemble_bilinear(m, U, V)
    M = HauntedMatrix(M, q0) # sparse matrix -> haunted sparse matrix

    # Parameters and time span to solve
    p = (U = U, V = V, vtk = vtk, l = l, iter = zeros(Int, 1), time = zeros(1000))
    tspan = (0.0, totalTime)

    odeFunction = ODEFunction(rhs!; mass_matrix = M)
    prob = ODEProblem(odeFunction, q0, tspan, p)
    @only_root println("Running implicit (dense) solve...")
    solve(
        prob,
        ImplicitEuler(; linsolve = PetscFactorization());
        callback = cbset,
        save_everystep = false,
    )
end

function timeintegration_impl_sparse(m, l, U, V, cbset, dmesh)
    # Init VTK
    rank = MPI.Comm_rank(BcubeMPI.get_comm(dmesh))
    vtk = VtkHandler(joinpath(out_dir, "linear_transport_impl_sparse"), dmesh)

    # Allocate vector of dofs (a HauntedVector)
    q0 = Bcube.allocate_dofs(U)

    # Mass matrix
    M = assemble_bilinear(m, U, V)
    M = HauntedMatrix(M, q0) # sparse matrix -> haunted sparse matrix

    # Parameters and time span to solve
    p = (U = U, V = V, vtk = vtk, l = l, iter = zeros(Int, 1), time = zeros(1000))
    tspan = (0.0, totalTime)

    # Jacobian cache
    @only_root println("computing jacobian cache...")
    _f! = (y, x) -> rhs!(y, x, p, 0.0)
    output = similar(q0)
    input = similar(q0)
    sparsity_pattern = Symbolics.jacobian_sparsity(_f!, output, input)
    jac = HauntedMatrix(Float64.(sparsity_pattern) + I, q0) # +I to ensure diag belongs to sparsity pattern
    colors = matrix_colors(jac)

    # Solve !
    odeFunction = ODEFunction(rhs!; mass_matrix = M, jac_prototype = jac, colorvec = colors)
    prob = ODEProblem(odeFunction, q0, tspan, p)
    @only_root println("Running implicit (sparse) solve...")
    solve(
        prob,
        ImplicitEuler(; linsolve = PetscFactorization());
        callback = cbset,
        save_everystep = false,
    )

    # BELOW IS FOR PROFILING
    if bench
        x = diff(p.time[5:p.iter[1]])
        println("Time by iteration : $(sum(x) / length(x)*1000) ms")

        # Profile.clear()
        # @profile solve(
        #     prob,
        #     ImplicitEuler(; linsolve = PetscFactorization());
        #     callback = cb_update,
        #     save_everystep = false,
        # )
    end
end

function rhs!(dQ, Q, p, t)
    # Update the FEFunction (ghost are updated via callback)
    u = FEFunction(p.U, Q; updateGhosts = false)

    # Compute linear forms
    dQ .= zero(eltype(Q))
    assemble_linear!(dQ, v -> p.l(v, u, t), p.V)
end

"""
Euler solver is not able to use mass matrix
"""
function f_expl!(dQ, Q, p, t)
    rhs!(dQ, Q, p, t)
    dQ .= p.M \ dQ
end

function run()
    # Init MPI
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    np = MPI.Comm_size(comm)

    init_petsc()

    # Output directory
    @only_root begin
        isdir(out_dir) || mkdir(out_dir)
        foreach(
            path -> rm(path; recursive = true),
            filter(
                filename -> startswith(basename(filename), "linear_transport_"),
                readdir(out_dir; join = true),
            ),
        )
    end

    # Then generate the mesh of a rectangle using Gmsh and read it
    tmp_path = joinpath(out_dir, "tmp.msh")
    @only_root println("building mesh...")
    @only_root BcubeGmsh.gen_rectangle_mesh(
        tmp_path,
        :quad;
        nx = nx,
        ny = ny,
        lx = lx,
        ly = ly,
        xc = 0.0,
        yc = 0.0,
        split_files = true,
        n_partitions = np,
        create_ghosts = true,
    )
    MPI.Barrier(comm) # wait for the mesh to be generated by proc 0

    @only_root println("reading mesh...")
    mesh = read_partitioned_msh(tmp_path, comm)

    # Define function space, FE spaces and the FEFunction
    fs = FunctionSpace(:Taylor, degree)
    U = TrialFESpace(fs, mesh, :discontinuous; cacheType = PetscCache)
    V = TestFESpace(U)

    # Define measures for cell and interior face integrations
    Γ = InteriorFaceDomain(mesh)
    Γ_in = BoundaryFaceDomain(mesh, "West")
    Γ_out = BoundaryFaceDomain(mesh, ("North", "East", "South"))

    dΩ = Measure(CellDomain(mesh), 2 * degree + 1)
    dΓ = Measure(Γ, 2 * degree + 1)
    dΓ_in = Measure(Γ_in, 2 * degree + 1)
    dΓ_out = Measure(Γ_out, 2 * degree + 1)

    nΓ = get_face_normals(Γ)
    nΓ_in = get_face_normals(Γ_in)
    nΓ_out = get_face_normals(Γ_out)

    # Integrals
    l_Ω(v, u) = ∫((c * u) ⋅ ∇(v))dΩ # Volumic convective term

    flux(u) = upwind ∘ (side⁻(u), side⁺(u), side⁻(nΓ))
    l_Γ(v, u) = ∫(flux(u) * jump(v))dΓ

    bc_in = t -> PhysicalFunction(x -> c .* cos(3 * x[2]) * sin(4 * t)) # flux
    l_Γ_in(v, t) = ∫(side⁻(bc_in(t)) ⋅ side⁻(nΓ_in) * side⁻(v))dΓ_in

    flux_out(u) = upwind ∘ (side⁻(u), 0.0, side⁻(nΓ_out))
    l_Γ_out(v, u) = ∫(flux_out(u) * side⁻(v))dΓ_out

    l(v, u, t) = l_Ω(v, u) - l_Γ(v, u) - l_Γ_in(v, t) - l_Γ_out(v, u)

    # Mass matrix
    m(u, v) = ∫(u ⋅ v)dΩ # Mass matrix

    # Callbacks
    cb_vtk = DiscreteCallback(
        always_true,
        integrator -> begin
            p = integrator.p
            u = FEFunction(p.U, integrator.u)
            append_vtk(p.vtk, u, integrator.t)

            @only_root @show integrator.t
        end;
        save_positions = (false, false),
    )
    cb_update = DiscreteCallback(
        always_true,
        integrator -> begin
            HauntedArrays.update_ghosts!(integrator.u)
        end;
        save_positions = (false, false),
    )
    cb_timer = DiscreteCallback(
        always_true,
        integrator -> begin
            p = integrator.p
            p.iter[1] += 1
            p.time[p.iter[1]] = time()
        end;
        save_positions = (false, false),
    )
    cbset = CallbackSet(cb_update, cb_vtk)
    if bench
        cbset = CallbackSet(cb_update, cb_timer)
    end
    # timeintegration_expl(m, l, U, V, cbset, mesh)
    # timeintegration_impl_dense(m, l, U, V, cbset, mesh)
    timeintegration_impl_sparse(m, l, U, V, cbset, mesh)

    @one_at_a_time println("Done running linear_transport!")
end

# Parameters (global variables for now)
const degree = 0 # Function-space degree (Taylor(0) = first order Finite Volume)
const c = [1.0, 0.0] # Convection velocity (must be a vector)
const CFL = 1 # 0.1 for degree 1
const nx = 11 # Number of nodes in the x-direction
const ny = 11 # Number of nodes in the y-direction
const lx = 2.0 # Domain width
const ly = 2.0 # Domain height
const totalTime = 10.0
const out_dir = joinpath(@__DIR__, "..", "..", "tmp", "linear_transport_par")
const bench = false
bench && @warn "bench set to true, no vtk export"

run()

end
