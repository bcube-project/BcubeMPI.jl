function rhs!(dQ, Q, p, t)
    dQ .= zero(eltype(dQ))
    q = (FEFunction(p.U, Q)...,)
    assemble_linear!(dQ, v -> p.l(q, v), p.V)
end

"""
    flux_Ω(q, v)

Compute volume residual using the lazy-operators approach
"""
flux_Ω(q, v) = _flux_Ω ∘ (q, map(∇, v))

function _flux_Ω(q, ∇v)
    ρ, ρu, ρE = q
    ∇λ_ρ, ∇λ_ρu, ∇λ_ρE = ∇v
    γ = stateInit.γ

    vel = ρu ./ ρ
    ρuu = ρu * transpose(vel)
    p = pressure(ρ, ρu, ρE, γ)

    flux_ρ  = ρu
    flux_ρu = ρuu + p * I
    flux_ρE = (ρE + p) .* vel

    return return ∇λ_ρ ⋅ flux_ρ + ∇λ_ρu ⊡ flux_ρu + ∇λ_ρE ⋅ flux_ρE
end

"""
    flux_Γ(q, v, n)

Flux at the interface is defined by a composition of two functions:
* the input states at face sides which are needed for the riemann flux
* `flux_roe` defines the Riemann flux (as usual)
"""
flux_Γ(q, v, n) = flux_roe ∘ (side⁻(q), side⁺(q), jump(v), side⁻(n))

"""
    flux_roe(q⁻, q⁺, δv, n)
"""
function flux_roe(q⁻, q⁺, δv, n)
    γ = stateInit.γ
    nx, ny = n
    ρ1, (ρu1, ρv1), ρE1 = q⁻
    ρ2, (ρu2, ρv2), ρE2 = q⁺
    δλ_ρ1, δλ_ρu1, δλ_ρE1 = δv

    ρ1 = max(eps(ρ1), ρ1)
    ρ2 = max(eps(ρ2), ρ2)

    # Closure
    u1 = ρu1 / ρ1
    v1 = ρv1 / ρ1
    u2 = ρu2 / ρ2
    v2 = ρv2 / ρ2
    p1 = pressure(ρ1, SA[ρu1, ρv1], ρE1, γ)
    p2 = pressure(ρ2, SA[ρu2, ρv2], ρE2, γ)

    H2 = (γ / (γ - 1)) * p2 / ρ2 + (u2 * u2 + v2 * v2) / 2.0
    H1 = (γ / (γ - 1)) * p1 / ρ1 + (u1 * u1 + v1 * v1) / 2.0

    R = √(ρ1 / ρ2)
    invR1 = 1.0 / (R + 1)
    uAv = (R * u1 + u2) * invR1
    vAv = (R * v1 + v2) * invR1
    Hav = (R * H1 + H2) * invR1
    cAv = √(abs((γ - 1) * (Hav - (uAv * uAv + vAv * vAv) / 2.0)))
    ecAv = (uAv * uAv + vAv * vAv) / 2.0

    λ1 = nx * uAv + ny * vAv
    λ3 = λ1 + cAv
    λ4 = λ1 - cAv

    d1 = ρ1 - ρ2
    d2 = ρ1 * u1 - ρ2 * u2
    d3 = ρ1 * v1 - ρ2 * v2
    d4 = ρE1 - ρE2

    # computation of the centered part of the flux
    flux_ρ  = nx * ρ2 * u2 + ny * ρ2 * v2
    flux_ρu = nx * p2 + flux_ρ * u2
    flux_ρv = ny * p2 + flux_ρ * v2
    flux_ρE = H2 * flux_ρ

    # Temp variables
    rc1 = (γ - 1) / cAv
    rc2 = (γ - 1) / cAv / cAv
    uq41 = ecAv / cAv + cAv / (γ - 1)
    uq42 = nx * uAv + ny * vAv

    fdc1 = max(λ1, 0.0) * (d1 + rc2 * (-ecAv * d1 + uAv * d2 + vAv * d3 - d4))
    fdc2 = max(λ1, 0.0) * ((nx * vAv - ny * uAv) * d1 + ny * d2 - nx * d3)
    fdc3 =
        max(λ3, 0.0) * (
            (-uq42 * d1 + nx * d2 + ny * d3) / 2.0 +
            rc1 * (ecAv * d1 - uAv * d2 - vAv * d3 + d4) / 2.0
        )
    fdc4 =
        max(λ4, 0.0) * (
            (uq42 * d1 - nx * d2 - ny * d3) / 2.0 +
            rc1 * (ecAv * d1 - uAv * d2 - vAv * d3 + d4) / 2.0
        )

    duv1 = fdc1 + (fdc3 + fdc4) / cAv
    duv2 = uAv * fdc1 + ny * fdc2 + (uAv / cAv + nx) * fdc3 + (uAv / cAv - nx) * fdc4
    duv3 = vAv * fdc1 - nx * fdc2 + (vAv / cAv + ny) * fdc3 + (vAv / cAv - ny) * fdc4
    duv4 =
        ecAv * fdc1 +
        (ny * uAv - nx * vAv) * fdc2 +
        (uq41 + uq42) * fdc3 +
        (uq41 - uq42) * fdc4

    flux_ρ  += duv1
    flux_ρu += duv2
    flux_ρv += duv3
    flux_ρE += duv4

    return (δλ_ρ1 ⋅ flux_ρ + δλ_ρu1 ⋅ SA[flux_ρu, flux_ρv] + δλ_ρE1 ⋅ flux_ρE)
end

"""
    flux_Γ_farfield(q, v, n)

Compute `Roe` flux on boundary face by imposing
`stateBcFarfield.u_in` on `side_p`
"""
flux_Γ_farfield(q, v, n) = flux_roe ∘ (side⁻(q), stateBcFarfield.u_inf, side⁻(v), side⁻(n))

"""
    flux_Γ_wall(q, v, n)
"""
flux_Γ_wall(q, v, n) = _flux_Γ_wall ∘ (side⁻(q), side⁻(v), side⁻(n))

function _flux_Γ_wall(q⁻, v⁻, n)
    γ = stateInit.γ
    ρ1, ρu1, ρE1 = q⁻
    λ_ρ1, λ_ρu1, λ_ρE1 = v⁻

    p1 = pressure(ρ1, ρu1, ρE1, γ)

    flux_ρ  = zero(ρ1)
    flux_ρu = p1 * n
    flux_ρE = zero(ρE1)

    return (λ_ρ1 ⋅ flux_ρ + λ_ρu1 ⋅ flux_ρu + λ_ρE1 ⋅ flux_ρE)
end

function init!(q, dΩ, initstate)
    AoA  = initstate.AoA
    Minf = initstate.M_inf
    Pinf = initstate.P_inf
    Tinf = initstate.T_inf
    r    = initstate.r_gas
    γ    = initstate.γ

    ρinf = Pinf / r / Tinf
    ainf = √(γ * r * Tinf)
    Vinf = Minf * ainf
    ρVxinf = ρinf * Vinf * cos(AoA)
    ρVyinf = ρinf * Vinf * sin(AoA)
    ρEinf = Pinf / (γ - 1) + 0.5 * ρinf * Vinf^2

    ρ0  = PhysicalFunction(x -> ρinf)
    ρu0 = PhysicalFunction(x -> SA[ρVxinf, ρVyinf])
    ρE0 = PhysicalFunction(x -> ρEinf)
    projection_l2!(q, (ρ0, ρu0, ρE0), dΩ)
    return nothing
end

always_true(args...) = true

function condition_steadystate(integrator, abstol, reltol, min_t)
    u_modified!(integrator, false)
    if DiffEqBase.isinplace(integrator.sol.prob)
        testval = first(get_tmp_cache(integrator))
        @. testval = (integrator.u - integrator.uprev) / (integrator.t - integrator.tprev)
    else
        testval = (integrator.u - integrator.uprev) / (integrator.t - integrator.tprev)
    end

    if typeof(integrator.u) <: Array
        any(
            abs(d) > abstol && abs(d) > reltol * abs(u) for (d, abstol, reltol, u) in
            zip(testval, Iterators.cycle(abstol), Iterators.cycle(reltol), integrator.u)
        ) && (return false)
    else
        any((abs.(testval) .> abstol) .& (abs.(testval) .> reltol .* abs.(integrator.u))) &&
            (return false)
    end

    if min_t === nothing
        return true
    else
        return integrator.t >= min_t
    end
end

function pressure(ρ::Number, ρu::AbstractVector, ρE::Number, γ)
    vel = ρu ./ ρ
    ρuu = ρu * transpose(vel)
    p = (γ - 1) * (ρE - tr(ρuu) / 2)
    return p
end

compute_Pᵢ(P, γ, M) = P * (1 + 0.5 * (γ - 1) * M^2)^(γ / (γ - 1))
compute_Tᵢ(T, γ, M) = T * (1 + 0.5 * (γ - 1) * M^2)

function bc_state_farfield(AoA, M, P, T, r, γ)
    a = √(γ * r * T)
    vn = M * a
    ρ = P / r / T
    ρu = SA[ρ * vn * cos(AoA), ρ * vn * sin(AoA)]
    ρE = P / (γ - 1) + 0.5 * ρ * vn^2
    return (ρ, ρu, ρE)
end

function pressure_coefficient(ρ, ρu, ρE)
    (pressure(ρ, ρu, ρE, stateInit.γ) - stateInit.P_inf) /
    (stateBcFarfield.Pᵢ_inf - stateInit.P_inf)
end
function mach(ρ, ρu, ρE)
    norm(ρu ./ ρ) / √(stateInit.γ * max(0.0, pressure(ρ, ρu, ρE, stateInit.γ) / ρ))
end
