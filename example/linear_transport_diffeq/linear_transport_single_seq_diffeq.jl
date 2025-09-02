module LinearTransport
println("Running single seq linear transport example...")
using Bcube, BcubeGmsh, BcubeVTK
using LinearAlgebra
using OrdinaryDiffEq
using Symbolics
using SparseDiffTools
include(joinpath(@__DIR__, "common.jl"))

function append_vtk(vtk, u::Bcube.AbstractFEFunction, t)
    # Write
    write_file(
        vtk.basename,
        vtk.mesh,
        Dict("u" => u),
        vtk.ite,
        t;
        collection_append = vtk.ite > 0,
    )

    # Update counter
    vtk.ite += 1
end

# Parameters
const degree = 0 # Function-space degree (Taylor(0) = first order Finite Volume)
const c = [1.0, 0.0] # Convection velocity (must be a vector)
const nite = 100 # Number of time iteration(s)
const CFL = 1 # 0.1 for degree 1
const nx = 20 # Number of nodes in the x-direction
const ny = 20 # Number of nodes in the y-direction
const lx = 2.0 # Domain width
const ly = 2.0 # Domain height
const totalTime = 2.0
const bench = false
bench && @warn "bench is set to true, vtk output are disabled"

function run()

    # Output directory
    out_dir = joinpath(@__DIR__, "..", "..", "tmp")
    isdir(out_dir) || mkdir(out_dir)
    # rm(out_dir * "/linear_transport*"; force = true)

    # Then generate the mesh of a rectangle using Gmsh and read it
    tmp_path = joinpath(out_dir, "tmp.msh")
    BcubeGmsh.gen_rectangle_mesh(
        tmp_path,
        :quad;
        nx = nx,
        ny = ny,
        lx = lx,
        ly = ly,
        xc = 0.0,
        yc = 0.0,
    )
    mesh = read_mesh(tmp_path)

    # We can now init our `VtkHandler`
    outputpath = joinpath(out_dir, "linear_transport_seq")
    rm(outputpath; recursive = true, force = true)
    mkpath(outputpath)
    vtk = VtkHandler(joinpath(outputpath, "result.pvd"), mesh)

    # Define function space, FE spaces and the FEFunction
    fs = FunctionSpace(:Taylor, degree)
    U = TrialFESpace(fs, mesh, :discontinuous)
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

    cb_vtk = DiscreteCallback(
        always_true,
        integrator -> begin
            p = integrator.p

            # Update the FEFunction
            u = FEFunction(p.U, integrator.u)
            append_vtk(vtk, u, integrator.t)
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
    cbset = CallbackSet(cb_vtk) # normal
    if bench
        cbset = CallbackSet(cb_timer) # for bench
    end

    # timeintegration_expl(m, l, U, V, cbset, dΩ)
    # timeintegration_impl_dense(m, l, U, V, cbset, dΩ)
    timeintegration_impl_sparse(m, l, U, V, cbset, dΩ)

    println("linear_transport done")
end

always_true(args...) = true

function timeintegration_expl(m, l, U, V, cbset, dΩ)
    Δt = CFL * min(lx / nx, ly / ny) / norm(c) # Time step
    M = assemble_bilinear(m, U, V)
    invM = inv(Matrix(M)) #WARNING : really expensive !!!

    tspan = (0.0, totalTime)
    p = (U = U, V = V, l = l, iter = zeros(Int, 1), time = zeros(1000), invM = invM)

    prob = ODEProblem(f_expl!, Bcube.allocate_dofs(U), tspan, p)
    println("Running solve explicit...")
    solve(prob, Euler(); dt = Δt, callback = cbset)
end

function timeintegration_impl_dense(m, l, U, V, cbset, dΩ)
    M = assemble_bilinear(m, U, V)
    M = Matrix(M) # sparse -> array

    tspan = (0.0, totalTime)
    p = (U = U, V = V, l = l, iter = zeros(Int, 1), time = zeros(1000))

    # odeFunction = ODEFunction(f!; mass_matrix = Bcube.build_mass_matrix(u, dΩ))
    odeFunction = ODEFunction(rhs!; mass_matrix = M)
    prob = ODEProblem(odeFunction, Bcube.allocate_dofs(U), tspan, p)
    println("Running solve (implicite dense)...")
    solve(prob, ImplicitEuler(); callback = cbset)
    # solve(prob, ImplicitEuler(; linsolve = SimpleLUFactorization()); callback = cbset)
end

function timeintegration_impl_sparse(m, l, U, V, cbset, dΩ)
    tspan = (0.0, totalTime)
    p = (U = U, V = V, l = l, iter = zeros(Int, 1), time = zeros(1000))

    println("computing jacobian cache...")
    _f! = (y, x) -> rhs!(y, x, p, 0.0)
    output = zeros(Bcube.get_ndofs(U))
    input = zeros(Bcube.get_ndofs(U))
    sparsity_pattern = Symbolics.jacobian_sparsity(_f!, output, input)
    jac = Float64.(sparsity_pattern)
    colors = matrix_colors(jac)

    u = FEFunction(U)

    odeFunction = ODEFunction(
        rhs!;
        mass_matrix = Bcube.build_mass_matrix(u, dΩ),
        jac_prototype = jac,
        colorvec = colors,
    )
    prob = ODEProblem(odeFunction, u.dofValues, tspan, p)
    println("Running solve (implicite sparse)...")
    solve(prob, ImplicitEuler(); callback = cbset)

    # For BENCH
    if bench
        x = diff(p.time[5:p.iter[1]])
        println("Time by iteration : $(sum(x) / length(x)*1000) ms")
    end
end

function f_expl!(dQ, Q, p, t)
    rhs!(dQ, Q, p, t)
    dQ .= p.invM * dQ
end

function rhs!(dQ, Q, p, t)
    # Update the FEFunction
    u = FEFunction(p.U, Q)

    # Compute linear forms
    # I don't know why, but assemble_linear!(dQ, v -> p.l(v, u, t), p.V)
    # leads to an incorrect result
    b = zero(Q)
    assemble_linear!(b, v -> p.l(v, u, t), p.V)
    dQ .= b
end

run()

end
