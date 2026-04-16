module PhaseField

using LinearAlgebra
using Printf
using Random
using Optim
using LineSearches

include("Roughness.jl")
using .Roughness

export compute_C, phase_field_energy, phase_field_gradient!,
       compute_volume, compute_volume_gradient, solve_volume_constrained,
       Roughness

# --- Potential and normalisation ---
W(u)  = u^2 * (u - 1)^2
dW(u) = 2.0 * u * (u - 1.0) * (2.0 * u - 1.0)

# --- Box mapping (sigmoid/logit) ---
# Maps unconstrained v to u ∈ (0, 1) to enforce box constraints.
function sigmoid(v::Float64)
    if v >= 0.0
        z = exp(-v)
        return 1.0 / (1.0 + z)
    else
        z = exp(v)
        return z / (1.0 + z)
    end
end
logit(u::Float64) = log(u / (1.0 - u))

# β[W] = 2∫₀¹ √W(u) du = 2∫₀¹ u(1-u) du = 1/3, so the perimeter prefactor is 1/β[W] = 3.
const perimeter_prefactor = 3.0

# --- Public helpers ---

"""
    compute_C(σ)

Capillary integral C(σ) = ∫_{-1/2}^{1/2} √(1-(2σt)²) dt = √(1-σ²)/2 + arcsin(σ)/(2σ).
The contact-angle–adhesion relation is σ = -cos θ (Young–Dupré).
"""
function compute_C(σ::Float64)
    if abs(σ) < 1e-6
        return 1.0 - σ^2 / 6.0
    else
        return 0.5 * sqrt(1.0 - σ^2) + asin(σ) / (2.0 * σ)
    end
end

# --- FEM phase-field energy ---
#
# Discretisation: P1 (piecewise-linear) triangular FEM on a regular periodic 2-D grid.
# Each pixel (i,j) is divided into two triangles by the diagonal from (i+1,j) to (i,j+1):
#
#   (i,j+1) ---- (i+1,j+1)
#      |        / |
#      | T1  /    |
#      |  /    T0 |
#   (i,j)  ---- (i+1,j)
#
#   Triangle 0 (lower-right): nodes (i,j), (i+1,j), (i,j+1)   — unit-pixel coord x₁+x₂ < 1
#   Triangle 1 (upper-left):  nodes (i+1,j+1), (i+1,j), (i,j+1) — unit-pixel coord x₁+x₂ > 1
#
# Centroid quadrature: one point per triangle at pixel-local (1/3,1/3) and (2/3,2/3),
# quad weight 0.5 each (integrates a triangle of area l²/2 with the centroid rule).
#
# Value interpolation at the centroid: arithmetic mean of the three corner nodes (N = 1/3 each).
# Gradient interpolation: constant per triangle (P1 gradients):
#   Triangle 0:  ∂u/∂x = (u[i+1,j] - u[i,j]) / l,   ∂u/∂y = (u[i,j+1] - u[i,j]) / l
#   Triangle 1:  ∂u/∂x = (u[i+1,j+1] - u[i,j+1]) / l, ∂u/∂y = (u[i+1,j+1] - u[i+1,j]) / l
#
# Grid spacings lx (x) and ly (y) may differ; square grids use lx = ly = l.
#
# Energy functional (paper eq. phase-field-approx-energy):
#
#   I_ε[u] = ∫_ω { g(x)·C(σ)·(1/β[W])·(ε|∇u|² + W(u)/ε) + 2σ·u } dx

"""
    phase_field_energy(u_vec, g, ε, lx, ly, σ_val, C_σ)

Evaluate the phase-field energy using P1 FEM with centroid quadrature.

# Arguments
- `u_vec`: phase-field values, length Nx*Ny (column-major, i.e. `u[i,j]` is the (i,j)-th node)
- `g`: gap field, size (Nx, Ny)
- `ε`: interface thickness
- `lx`, `ly`: grid spacings in x and y (pass `l, l` for a square grid)
- `σ_val`: adhesion coefficient σ = -cos θ
- `C_σ`: precomputed C(σ); pass `compute_C(σ_val)`
"""
function phase_field_energy(u_vec::Vector{Float64}, g::Matrix{Float64},
                             ε::Float64, lx::Float64, ly::Float64,
                             σ_val::Float64, C_σ::Float64)
    Nx, Ny = size(g)
    u = reshape(u_vec, Nx, Ny)
    energy  = 0.0
    inv_ε   = 1.0 / ε
    inv_lx  = 1.0 / lx
    inv_ly  = 1.0 / ly
    pf      = perimeter_prefactor
    aw      = 0.5 * lx * ly

    for j in 1:Ny
        jp1 = j < Ny ? j + 1 : 1
        for i in 1:Nx
            ip1 = i < Nx ? i + 1 : 1

            u00 = u[i, j];    u10 = u[ip1, j];    u01 = u[i, jp1];    u11 = u[ip1, jp1]
            g00 = g[i, j];    g10 = g[ip1, j];    g01 = g[i, jp1];    g11 = g[ip1, jp1]

            u_q0  = (u00 + u10 + u01) / 3.0
            g_q0  = (g00 + g10 + g01) / 3.0
            gx_q0 = (u10 - u00) * inv_lx
            gy_q0 = (u01 - u00) * inv_ly
            e_q0  = g_q0 * C_σ * pf * (ε * (gx_q0^2 + gy_q0^2) + W(u_q0) * inv_ε) +
                    (g_q0 > 0.0 ? 2.0 * σ_val * u_q0 : 0.0)

            u_q1  = (u11 + u10 + u01) / 3.0
            g_q1  = (g11 + g10 + g01) / 3.0
            gx_q1 = (u11 - u01) * inv_lx
            gy_q1 = (u11 - u10) * inv_ly
            e_q1  = g_q1 * C_σ * pf * (ε * (gx_q1^2 + gy_q1^2) + W(u_q1) * inv_ε) +
                    (g_q1 > 0.0 ? 2.0 * σ_val * u_q1 : 0.0)

            energy += aw * (e_q0 + e_q1)
        end
    end
    return energy
end

phase_field_energy(u_vec::Vector{Float64}, g::Matrix{Float64},
                   ε::Float64, l::Float64, σ_val::Float64, C_σ::Float64) =
    phase_field_energy(u_vec, g, ε, l, l, σ_val, C_σ)

"""
    phase_field_gradient!(G_vec, u_vec, g, ε, lx, ly, σ_val, C_σ)

Compute the gradient of `phase_field_energy` w.r.t. `u_vec` via the exact adjoint of the P1 FEM.
Result is accumulated into `G_vec` (which is first zeroed).
Pass `l, l` for `lx, ly` on a square grid.
"""
function phase_field_gradient!(G_vec::Vector{Float64}, u_vec::Vector{Float64},
                                g::Matrix{Float64}, ε::Float64,
                                lx::Float64, ly::Float64,
                                σ_val::Float64, C_σ::Float64)
    Nx, Ny = size(g)
    u = reshape(u_vec, Nx, Ny)
    G = reshape(G_vec, Nx, Ny)
    fill!(G, 0.0)

    inv_ε  = 1.0 / ε
    inv_lx = 1.0 / lx
    inv_ly = 1.0 / ly
    pf     = perimeter_prefactor
    aw     = 0.5 * lx * ly

    for j in 1:Ny
        jp1 = j < Ny ? j + 1 : 1
        for i in 1:Nx
            ip1 = i < Nx ? i + 1 : 1

            u00 = u[i,j]; u10 = u[ip1,j]; u01 = u[i,jp1]; u11 = u[ip1,jp1]
            g00 = g[i,j]; g10 = g[ip1,j]; g01 = g[i,jp1]; g11 = g[ip1,jp1]

            u_q0  = (u00 + u10 + u01) / 3.0
            g_q0  = (g00 + g10 + g01) / 3.0
            gx_q0 = (u10 - u00) * inv_lx
            gy_q0 = (u01 - u00) * inv_ly

            de_du0  = g_q0 * C_σ * pf * inv_ε * dW(u_q0) + (g_q0 > 0.0 ? 2.0 * σ_val : 0.0)
            de_dgx0 = g_q0 * C_σ * pf * ε * 2.0 * gx_q0
            de_dgy0 = g_q0 * C_σ * pf * ε * 2.0 * gy_q0

            v0 = aw / 3.0 * de_du0
            G[i,j] += v0;  G[ip1,j] += v0;  G[i,jp1] += v0

            s = aw * inv_lx * de_dgx0
            G[i,j] -= s;  G[ip1,j] += s

            s = aw * inv_ly * de_dgy0
            G[i,j] -= s;  G[i,jp1] += s

            u_q1  = (u11 + u10 + u01) / 3.0
            g_q1  = (g11 + g10 + g01) / 3.0
            gx_q1 = (u11 - u01) * inv_lx
            gy_q1 = (u11 - u10) * inv_ly

            de_du1  = g_q1 * C_σ * pf * inv_ε * dW(u_q1) + (g_q1 > 0.0 ? 2.0 * σ_val : 0.0)
            de_dgx1 = g_q1 * C_σ * pf * ε * 2.0 * gx_q1
            de_dgy1 = g_q1 * C_σ * pf * ε * 2.0 * gy_q1

            v1 = aw / 3.0 * de_du1
            G[ip1,jp1] += v1;  G[ip1,j] += v1;  G[i,jp1] += v1

            s = aw * inv_lx * de_dgx1
            G[i,jp1] -= s;  G[ip1,jp1] += s

            s = aw * inv_ly * de_dgy1
            G[ip1,j] -= s;  G[ip1,jp1] += s
        end
    end
    return G_vec
end

phase_field_gradient!(G_vec::Vector{Float64}, u_vec::Vector{Float64},
                      g::Matrix{Float64}, ε::Float64, l::Float64,
                      σ_val::Float64, C_σ::Float64) =
    phase_field_gradient!(G_vec, u_vec, g, ε, l, l, σ_val, C_σ)

"""
    compute_volume(u_vec, g, lx, ly)

Approximate V[u] = ∫_ω u(x)·g(x) dx via the nodal quadrature rule (cell-centred sum).
For spatially uniform g this is exact relative to the FEM quadrature.
Pass `l, l` for `lx, ly` on a square grid.
"""
function compute_volume(u_vec::Vector{Float64}, g::Matrix{Float64}, lx::Float64, ly::Float64)
    Nx, Ny = size(g)
    u = reshape(u_vec, Nx, Ny)
    vol = 0.0
    for j in 1:Ny, i in 1:Nx
        vol += u[i, j] * g[i, j] * lx * ly
    end
    return vol
end

compute_volume(u_vec::Vector{Float64}, g::Matrix{Float64}, l::Float64) =
    compute_volume(u_vec, g, l, l)

"""
    compute_volume_gradient(g, lx, ly)

Return the gradient of `compute_volume` w.r.t. `u_vec`: ∂V/∂u[k] = g[k]·lx·ly.
Pass `l, l` for `lx, ly` on a square grid.
"""
function compute_volume_gradient(g::Matrix{Float64}, lx::Float64, ly::Float64)
    return vec(g) .* (lx * ly)
end

compute_volume_gradient(g::Matrix{Float64}, l::Float64) =
    compute_volume_gradient(g, l, l)

"""
    solve_volume_constrained(energy_fn, gradient_fn!, volume_fn, vol_grad, u0, vol_target;
                             tol_h=1e-7, max_outer=30, inner_iter=500,
                             c_init=10.0, verbose=false)

Minimise `energy_fn(u)` subject to `volume_fn(u) = vol_target` and `0 ≤ u ≤ 1`
via an augmented-Lagrangian outer loop with box-constrained L-BFGS as the inner solver.

The augmented Lagrangian is:

  L(u; λ, c) = energy_fn(u) + λ·(V[u]−V*) + c/2·(V[u]−V*)²

whose gradient is:

  ∂L/∂u = gradient_fn!(G, u) + (λ + c·(V−V*)) · vol_grad

# Arguments
- `energy_fn`   : `u::Vector{Float64} → Float64` — objective energy
- `gradient_fn!`: `(G, u)` — fills G with ∂energy/∂u (in-place, G is zeroed by the caller)
- `volume_fn`   : `u → Float64` — volume functional V[u]
- `vol_grad`    : `Vector{Float64}` — gradient of the volume w.r.t. u (∂V/∂u[k] = g[k]·l²)
- `u0`          : initial phase-field (will be clamped to (0,1) internally)
- `vol_target`  : target volume V*

# Keyword arguments
- `tol_h`      : convergence tolerance on |V[u] − V*|  (default 1e-7)
- `max_outer`  : maximum augmented-Lagrangian outer iterations  (default 30)
- `inner_iter` : maximum L-BFGS iterations per inner solve  (default 500)
- `c_init`     : initial quadratic-penalty coefficient  (default 10.0)
- `verbose`    : print convergence table if true  (default false)

# Returns
`(u, λ)` where `u` is the optimal phase-field and `λ` is the converged Lagrange
multiplier (equal to the Laplace pressure at convergence).
"""
function solve_volume_constrained(
    energy_fn,
    gradient_fn!,
    volume_fn,
    vol_grad::Vector{Float64},
    u0::Vector{Float64},
    vol_target::Float64;
    contact    = nothing,   # Bool vector, length N; true = node is in contact, u forced to 0
    tol_h      = 1e-7,
    max_outer  = 30,
    inner_iter = 500,
    c_init     = 10.0,
    verbose    = false,
)
    has_contact = !isnothing(contact) && any(contact)
    eps_clamp   = 1e-10
    v           = logit.(clamp.(u0, eps_clamp, 1.0 - eps_clamp))
    has_contact && (v[contact] .= logit(eps_clamp))   # initialise contact nodes to u ≈ 0
    Gu     = similar(v)
    λ      = 0.0
    c      = Float64(c_init)
    h_prev = Inf

    if verbose
        @printf("%-6s │ %-14s │ %-12s │ %-12s\n", "outer", "energy", "|V−V*|", "λ")
        println("─"^52)
    end

    for outer in 1:max_outer
        λk, ck = λ, c

        function f(v_k)
            u_k = sigmoid.(v_k)
            has_contact && (u_k[contact] .= 0.0)
            V   = volume_fn(u_k)
            h   = V - vol_target
            return energy_fn(u_k) + λk * h + 0.5 * ck * h^2
        end

        function g!(Gv, v_k)
            u_k = sigmoid.(v_k)
            has_contact && (u_k[contact] .= 0.0)
            V   = volume_fn(u_k)
            h   = V - vol_target
            mul = λk + ck * h

            gradient_fn!(Gu, u_k)
            Gu .+= mul .* vol_grad
            has_contact && (Gu[contact] .= 0.0)

            # Gv = Gu * du/dv; du/dv = sigmoid'(v) = u*(1-u)
            for i in eachindex(Gv)
                Gv[i] = Gu[i] * u_k[i] * (1.0 - u_k[i])
            end
            has_contact && (Gv[contact] .= 0.0)
            return Gv
        end

        res = optimize(f, g!, v, LBFGS(linesearch=LineSearches.BackTracking()),
                       Optim.Options(iterations = inner_iter, g_tol = 1e-6))
        v   = Optim.minimizer(res)
        u   = sigmoid.(v)
        has_contact && (u[contact] .= 0.0)

        V = volume_fn(u)
        h = V - vol_target
        E = energy_fn(u)

        abs(h) < tol_h && return u, λ

        λ += c * h
        abs(h) > 0.25 * abs(h_prev) && (c *= 2.0)
        h_prev = h

        if verbose
            @printf("%-6d │ %-14.4e │ %-12.4e │ %-12.4e\n", outer, E, abs(h), λ)
        end
    end

    @warn "solve_volume_constrained: augmented Lagrangian did not converge to tol_h = $tol_h"
    u = sigmoid.(v)
    has_contact && (u[contact] .= 0.0)
    return u, λ
end

end # module
