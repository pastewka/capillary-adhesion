module PhaseField

using LinearAlgebra
using Printf
using Random

include("Roughness.jl")
using .Roughness

export compute_C, phase_field_energy, phase_field_gradient!,
    compute_volume, compute_volume_gradient, solve_volume_constrained,
    square_initial, Roughness

# --- Potential and normalisation ---
W(u) = u^2 * (u - 1)^2
dW(u) = 2.0 * u * (u - 1.0) * (2.0 * u - 1.0)

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
    energy = 0.0
    inv_ε = 1.0 / ε
    inv_lx = 1.0 / lx
    inv_ly = 1.0 / ly
    pf = perimeter_prefactor
    aw = 0.5 * lx * ly

    for j in 1:Ny
        jp1 = j < Ny ? j + 1 : 1
        for i in 1:Nx
            ip1 = i < Nx ? i + 1 : 1

            u00 = u[i, j]
            u10 = u[ip1, j]
            u01 = u[i, jp1]
            u11 = u[ip1, jp1]
            g00 = g[i, j]
            g10 = g[ip1, j]
            g01 = g[i, jp1]
            g11 = g[ip1, jp1]

            u_q0 = (u00 + u10 + u01) / 3.0
            g_q0 = (g00 + g10 + g01) / 3.0
            gx_q0 = (u10 - u00) * inv_lx
            gy_q0 = (u01 - u00) * inv_ly
            # Wetting term 2σu is integrated over all of ω; in contact regions
            # the `contact` mask of solve_volume_constrained forces u=0, so the
            # contribution from those nodes vanishes naturally.
            e_q0 = g_q0 * C_σ * pf * (ε * (gx_q0^2 + gy_q0^2) + W(u_q0) * inv_ε) +
                   2.0 * σ_val * u_q0

            u_q1 = (u11 + u10 + u01) / 3.0
            g_q1 = (g11 + g10 + g01) / 3.0
            gx_q1 = (u11 - u01) * inv_lx
            gy_q1 = (u11 - u10) * inv_ly
            e_q1 = g_q1 * C_σ * pf * (ε * (gx_q1^2 + gy_q1^2) + W(u_q1) * inv_ε) +
                   2.0 * σ_val * u_q1

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

    inv_ε = 1.0 / ε
    inv_lx = 1.0 / lx
    inv_ly = 1.0 / ly
    pf = perimeter_prefactor
    aw = 0.5 * lx * ly

    for j in 1:Ny
        jp1 = j < Ny ? j + 1 : 1
        for i in 1:Nx
            ip1 = i < Nx ? i + 1 : 1

            u00 = u[i, j]
            u10 = u[ip1, j]
            u01 = u[i, jp1]
            u11 = u[ip1, jp1]
            g00 = g[i, j]
            g10 = g[ip1, j]
            g01 = g[i, jp1]
            g11 = g[ip1, jp1]

            u_q0 = (u00 + u10 + u01) / 3.0
            g_q0 = (g00 + g10 + g01) / 3.0
            gx_q0 = (u10 - u00) * inv_lx
            gy_q0 = (u01 - u00) * inv_ly

            de_du0 = g_q0 * C_σ * pf * inv_ε * dW(u_q0) + 2.0 * σ_val
            de_dgx0 = g_q0 * C_σ * pf * ε * 2.0 * gx_q0
            de_dgy0 = g_q0 * C_σ * pf * ε * 2.0 * gy_q0

            v0 = aw / 3.0 * de_du0
            G[i, j] += v0
            G[ip1, j] += v0
            G[i, jp1] += v0

            s = aw * inv_lx * de_dgx0
            G[i, j] -= s
            G[ip1, j] += s

            s = aw * inv_ly * de_dgy0
            G[i, j] -= s
            G[i, jp1] += s

            u_q1 = (u11 + u10 + u01) / 3.0
            g_q1 = (g11 + g10 + g01) / 3.0
            gx_q1 = (u11 - u01) * inv_lx
            gy_q1 = (u11 - u10) * inv_ly

            de_du1 = g_q1 * C_σ * pf * inv_ε * dW(u_q1) + 2.0 * σ_val
            de_dgx1 = g_q1 * C_σ * pf * ε * 2.0 * gx_q1
            de_dgy1 = g_q1 * C_σ * pf * ε * 2.0 * gy_q1

            v1 = aw / 3.0 * de_du1
            G[ip1, jp1] += v1
            G[ip1, j] += v1
            G[i, jp1] += v1

            s = aw * inv_lx * de_dgx1
            G[i, jp1] -= s
            G[ip1, jp1] += s

            s = aw * inv_ly * de_dgy1
            G[ip1, j] -= s
            G[ip1, jp1] += s
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

Compute V[u] = ∫_ω u(x)·g(x) dx via the same P1 FEM centroid quadrature used by
`phase_field_energy`: each pixel is split into two triangles along the
(i+1,j)→(i,j+1) diagonal, and each triangle contributes `area · u_q · g_q`
with centroid values `u_q = (u_a + u_b + u_c)/3`, `g_q = (g_a + g_b + g_c)/3`.

For spatially uniform `g` this reduces to the nodal sum `Σ u_ij · g · lx·ly`
(exact under periodicity), so the earlier uniform-`g` identity is preserved.
For spatially varying `g` the FEM and nodal rules differ at O(lx,ly).
Pass `l, l` for `lx, ly` on a square grid.
"""
function compute_volume(u_vec::Vector{Float64}, g::Matrix{Float64}, lx::Float64, ly::Float64)
    Nx, Ny = size(g)
    u = reshape(u_vec, Nx, Ny)
    vol = 0.0
    aw  = 0.5 * lx * ly   # area per triangle
    for j in 1:Ny
        jp1 = j < Ny ? j + 1 : 1
        for i in 1:Nx
            ip1 = i < Nx ? i + 1 : 1
            u00 = u[i, j];  u10 = u[ip1, j];  u01 = u[i, jp1];  u11 = u[ip1, jp1]
            g00 = g[i, j];  g10 = g[ip1, j];  g01 = g[i, jp1];  g11 = g[ip1, jp1]
            u_q0 = (u00 + u10 + u01) / 3.0
            g_q0 = (g00 + g10 + g01) / 3.0
            u_q1 = (u11 + u10 + u01) / 3.0
            g_q1 = (g11 + g10 + g01) / 3.0
            vol += aw * (u_q0 * g_q0 + u_q1 * g_q1)
        end
    end
    return vol
end

compute_volume(u_vec::Vector{Float64}, g::Matrix{Float64}, l::Float64) =
    compute_volume(u_vec, g, l, l)

"""
    compute_volume_gradient(g, lx, ly)

Return the gradient of `compute_volume` w.r.t. `u_vec`. With the FEM rule
`V = Σ_T area_T · u_q · g_q`, node k receives `(area_T/3) · g_q` from each of
the 6 triangles incident to it, so the gradient is a locally smoothed version
of `g`. For uniform `g` this collapses to `g · lx·ly` (identical to the old
nodal gradient). Pass `l, l` for `lx, ly` on a square grid.
"""
function compute_volume_gradient(g::Matrix{Float64}, lx::Float64, ly::Float64)
    Nx, Ny = size(g)
    vg  = zeros(Nx, Ny)
    aw3 = (lx * ly) / 6.0   # triangle-area / 3
    for j in 1:Ny
        jp1 = j < Ny ? j + 1 : 1
        for i in 1:Nx
            ip1  = i < Nx ? i + 1 : 1
            g_q0 = (g[i, j]      + g[ip1, j] + g[i, jp1]) / 3.0
            g_q1 = (g[ip1, jp1]  + g[ip1, j] + g[i, jp1]) / 3.0
            # Triangle 0 contributes to (i,j), (ip1,j), (i,jp1)
            vg[i,   j]   += aw3 * g_q0
            vg[ip1, j]   += aw3 * g_q0
            vg[i,   jp1] += aw3 * g_q0
            # Triangle 1 contributes to (ip1,jp1), (ip1,j), (i,jp1)
            vg[ip1, jp1] += aw3 * g_q1
            vg[ip1, j]   += aw3 * g_q1
            vg[i,   jp1] += aw3 * g_q1
        end
    end
    return vec(vg)
end

compute_volume_gradient(g::Matrix{Float64}, l::Float64) =
    compute_volume_gradient(g, l, l)

"""
    _project_feasible(y, a, V_target, contact; tol=1e-14)

Project `y` onto the feasible polytope `F = {u ∈ [0,1]^N : aᵀu = V*, u[contact] = 0}`.
Assumes `a ≥ 0` elementwise (which holds for `vol_grad = compute_volume_gradient(g, …)`
with `g ≥ 0`). Because contact nodes satisfy `g_i = 0 ⇒ a_i = 0`, they automatically drop
out of the linear constraint; we only need to zero them explicitly.

The projection has the closed form `u_i = clip(y_i − μ·a_i, 0, 1)` where the multiplier
`μ` is the unique root of `Σ aᵢ · clip(yᵢ − μ·aᵢ, 0, 1) = V*`. Since `a ≥ 0` the
left-hand side is non-increasing in `μ`, so we bracket and bisect.
"""
function _project_feasible(y::Vector{Float64}, a::Vector{Float64},
                           V_target::Float64, contact; tol::Float64=1e-14)
    N = length(y)
    u = similar(y)
    has_contact = !isnothing(contact) && any(contact)

    Φ = μ -> begin
        s = 0.0
        @inbounds for i in 1:N
            (has_contact && contact[i]) && continue
            s += a[i] * clamp(y[i] - μ * a[i], 0.0, 1.0)
        end
        s - V_target
    end

    # Bracket: Φ(-∞) = Σ aᵢ·1 − V* ≥ 0, Φ(+∞) = -V* ≤ 0. Expand geometrically.
    μ_lo, μ_hi = -1.0, 1.0
    Φ_lo = Φ(μ_lo)
    while Φ_lo < 0.0
        μ_lo *= 2.0
        abs(μ_lo) > 1e20 && error("_project_feasible: cannot bracket μ (is V_target ≤ Σaᵢ?)")
        Φ_lo = Φ(μ_lo)
    end
    Φ_hi = Φ(μ_hi)
    while Φ_hi > 0.0
        μ_hi *= 2.0
        abs(μ_hi) > 1e20 && error("_project_feasible: cannot bracket μ (is V_target ≥ 0?)")
        Φ_hi = Φ(μ_hi)
    end

    # Bisect. 60 steps squeeze a `1e6` bracket below eps(Float64).
    for _ in 1:60
        μ_mid = 0.5 * (μ_lo + μ_hi)
        Φ_mid = Φ(μ_mid)
        if Φ_mid > 0.0
            μ_lo = μ_mid
        else
            μ_hi = μ_mid
        end
        abs(Φ_mid) < tol && break
    end
    μ = 0.5 * (μ_lo + μ_hi)

    @inbounds for i in 1:N
        u[i] = (has_contact && contact[i]) ? 0.0 : clamp(y[i] - μ * a[i], 0.0, 1.0)
    end
    return u
end

"""
    _lbfgs_direction(g, s_hist, y_hist, ρ_hist)

Two-loop recursion: returns `d = -H·g` where `H` is the L-BFGS Hessian approximation
from the most-recent-first histories `(s_hist, y_hist, ρ_hist)`. The initial scaling
is Nocedal–Wright's `γ = ⟨s,y⟩ / ⟨y,y⟩` on the newest pair (identity if history empty).
"""
function _lbfgs_direction(g::Vector{Float64},
                          s_hist::Vector{Vector{Float64}},
                          y_hist::Vector{Vector{Float64}},
                          ρ_hist::Vector{Float64})
    q = copy(g)
    n = length(s_hist)
    α = Vector{Float64}(undef, n)
    @inbounds for k in 1:n            # newest → oldest
        α[k] = ρ_hist[k] * dot(s_hist[k], q)
        q .-= α[k] .* y_hist[k]
    end
    if n > 0
        γ = dot(s_hist[1], y_hist[1]) / dot(y_hist[1], y_hist[1])
        q .*= γ
    end
    @inbounds for k in n:-1:1          # oldest → newest
        β = ρ_hist[k] * dot(y_hist[k], q)
        q .+= (α[k] - β) .* s_hist[k]
    end
    return -q
end

"""
    _kkt_residual(Gp, u, contact; tol_box=1e-12)

First-order KKT residual on the polytope. Components where `u` sits on an active box
face *and* the projected gradient `Gp` points *into* that face are zeroed out (the bound
is working, no improvement possible). The returned vector's norm is the convergence
criterion.
"""
function _kkt_residual(Gp::Vector{Float64}, u::Vector{Float64}, contact;
                       tol_box::Float64=1e-12)
    N = length(Gp)
    r = similar(Gp)
    has_contact = !isnothing(contact) && any(contact)
    @inbounds for i in 1:N
        if has_contact && contact[i]
            r[i] = 0.0
        elseif u[i] <= tol_box && Gp[i] >= 0.0
            r[i] = 0.0
        elseif u[i] >= 1.0 - tol_box && Gp[i] <= 0.0
            r[i] = 0.0
        else
            r[i] = Gp[i]
        end
    end
    return r
end

"""
    solve_volume_constrained(energy_fn, gradient_fn!, volume_fn, vol_grad, u0, vol_target;
                             contact=nothing,
                             g_tol=1e-7,
                             max_iter=500,
                             lbfgs_memory=10,
                             verbose=false)

Minimise `energy_fn(u)` subject to `⟨vol_grad, u⟩ = vol_target` and `0 ≤ u ≤ 1` by
**projected L-BFGS** on the feasible polytope `F = [0,1]^N ∩ {aᵀu = V*}`.

The volume functional is linear in `u` (`V[u] = ⟨a, u⟩` with `a = vol_grad`), so the
Lagrange multiplier is available in closed form at every iterate,
`λ = ⟨a, ∇E(u)⟩ / ⟨a, a⟩`, and no outer-loop / penalty-ramping / augmented-Lagrangian
scaffolding is needed. The tangent-projected gradient `Gₚ = ∇E − λ·a` drives the L-BFGS
update, and each trial step is projected back onto `F` (via `_project_feasible`, closed
form for a linear equality plus box bounds) so the iterates are *exactly feasible*.

# Arguments
- `energy_fn`    : `u → Float64` — objective energy.
- `gradient_fn!` : `(G, u)` — fills `G` with `∂energy/∂u` (the callee zeroes `G` first).
- `volume_fn`    : kept for signature stability; unused by the projected solver since
                   `⟨vol_grad, u⟩` is held at `vol_target` by projection.
- `vol_grad`     : gradient of the linear volume functional `V[u] = ⟨vol_grad, u⟩`;
                   use `compute_volume_gradient(g, lx, ly)`.
- `u0`           : initial phase-field (will be projected onto `F` before the loop).
- `vol_target`   : target volume `V*`.

# Keyword arguments
- `contact`      : optional `AbstractVector{Bool}` of length `N` marking nodes forced
                   to `u = 0` (e.g. solid–solid contact where `g = 0`).
- `g_tol`        : convergence tolerance on the per-node ∞-norm of the
                   tangent-projected gradient (box-active components masked).
                   The ∞-norm is scale-invariant in `N`, so the same `g_tol`
                   value is meaningful across grid sizes  (default `1e-5`).
- `max_iter`     : maximum outer iterations  (default `500`).
- `lbfgs_memory` : number of `(s, y)` curvature pairs retained  (default `10`).
- `verbose`      : print a per-iteration table if `true`  (default `false`).

# Returns
`(u, λ, residual)` where `u` is the optimal phase-field, `λ` is the Laplace-pressure
multiplier (computed in closed form at the last iterate), and `residual` is the
final ∞-norm of the box-masked tangent-projected gradient. By construction
`⟨vol_grad, u⟩ = vol_target` to within projection precision (≈ 1e-14).
"""
function solve_volume_constrained(
    energy_fn,
    gradient_fn!,
    volume_fn,   # unused; retained for backward-compat call signature
    vol_grad::Vector{Float64},
    u0::Vector{Float64},
    vol_target::Float64;
    contact=nothing,
    g_tol::Float64=1e-5,
    max_iter::Int=500,
    lbfgs_memory::Int=10,
    verbose::Bool=false,
)
    a   = vol_grad
    aTa = dot(a, a)
    @assert aTa > 0 "vol_grad is identically zero; no volume constraint to enforce"
    has_contact = !isnothing(contact) && any(contact)

    # Feasible starting iterate
    u = _project_feasible(u0, a, vol_target, contact)
    N = length(u)
    Gu = zeros(N)
    gradient_fn!(Gu, u)
    λ = dot(a, Gu) / aTa
    Gp = Gu .- λ .* a
    has_contact && (Gp[contact] .= 0.0)

    E = energy_fn(u)
    r_norm = maximum(abs, _kkt_residual(Gp, u, contact))

    # Most-recent-first circular buffer of curvature pairs.
    s_hist = Vector{Vector{Float64}}()
    y_hist = Vector{Vector{Float64}}()
    ρ_hist = Float64[]

    if verbose
        @printf("%-5s │ %-14s │ %-12s │ %-12s │ %-6s\n", "iter", "energy", "‖Gₚ‖_free", "λ", "α")
        println("─"^64)
        @printf("%-5d │ %-14.6e │ %-12.4e │ %-12.4e │ %-6s\n", 0, E, r_norm, λ, "-")
    end

    r_norm < g_tol && return u, λ, r_norm

    for iter in 1:max_iter
        # L-BFGS search direction, then re-tangent-project (small numerical drift).
        d = _lbfgs_direction(Gp, s_hist, y_hist, ρ_hist)
        d .-= (dot(a, d) / aTa) .* a
        has_contact && (d[contact] .= 0.0)

        gd = dot(Gp, d)
        if gd >= 0.0
            # Not a descent direction (stale curvature): reset and try -Gp.
            empty!(s_hist); empty!(y_hist); empty!(ρ_hist)
            d  = -Gp
            gd = dot(Gp, d)
        end

        # Backtracking line search with projection. The projection arc changes d
        # after clipping, so we check the plain Armijo condition E_new ≤ E+c1·α·gd.
        # If the L-BFGS direction fails (e.g. the active set just changed and the
        # stored curvature is stale), reset history and retry with scaled steepest
        # descent before giving up.
        α_init   = 1.0
        c1       = 1e-4
        u_new    = u
        E_new    = E
        accepted = false

        for attempt in 1:2
            α = α_init
            for _ in 1:40
                u_try = _project_feasible(u .+ α .* d, a, vol_target, contact)
                E_try = energy_fn(u_try)
                if E_try <= E + c1 * α * gd
                    u_new, E_new = u_try, E_try
                    accepted = true
                    break
                end
                α *= 0.5
            end
            accepted && break
            # Retry with steepest descent, scaled so the first trial step has
            # a reasonable magnitude instead of possibly blowing past the minimum.
            empty!(s_hist); empty!(y_hist); empty!(ρ_hist)
            d      = -Gp
            gd     = dot(Gp, d)
            gp_n   = norm(Gp)
            α_init = gp_n > 0 ? 1.0 / gp_n : 1.0
        end
        if !accepted
            verbose && @warn "line search failed at iter $iter (residual $r_norm)"
            break
        end

        # New gradient and tangent projection.
        Gu_new = zeros(N)
        gradient_fn!(Gu_new, u_new)
        λ_new  = dot(a, Gu_new) / aTa
        Gp_new = Gu_new .- λ_new .* a
        has_contact && (Gp_new[contact] .= 0.0)

        # Update L-BFGS history (only if the curvature pair is well-defined).
        s = u_new .- u
        y = Gp_new .- Gp
        sy = dot(s, y)
        if sy > 1e-12 * (norm(s) * norm(y) + eps())
            pushfirst!(s_hist, s); pushfirst!(y_hist, y); pushfirst!(ρ_hist, 1.0 / sy)
            if length(s_hist) > lbfgs_memory
                pop!(s_hist); pop!(y_hist); pop!(ρ_hist)
            end
        end

        u, Gu, Gp, λ, E = u_new, Gu_new, Gp_new, λ_new, E_new
        r_norm = maximum(abs, _kkt_residual(Gp, u, contact))

        verbose && @printf("%-5d │ %-14.6e │ %-12.4e │ %-12.4e │ %-6.2e\n", iter, E, r_norm, λ, α)

        r_norm < g_tol && return u, λ, r_norm
    end

    @warn "solve_volume_constrained: projected L-BFGS did not converge (‖Gₚ‖_free = $r_norm > $g_tol)"
    return u, λ, r_norm
end

"""
    square_initial(Nx, Ny, l, ε, vol_fraction; clamp_range=(0.02, 0.98))

Smooth square-droplet initial condition on a periodic `Nx × Ny` grid with spacing
`l`. The square is centred in the domain with side length
`√(vol_fraction · Nx · Ny · l²)` so its area equals `vol_fraction · |ω|`; the
edges are smoothed by `tanh` with width `ε`.

`clamp_range = (lo, hi)` clamps the returned values away from the saturation endpoints
`0` and `1`. Not required by the projected-L-BFGS solver (which handles the box
constraint natively), but retained as a mild safeguard for users who plug the output into
other optimisers. Pass `clamp_range = nothing` to disable the clamp.

Returns a `Vector{Float64}` of length `Nx·Ny` (column-major).
"""
function square_initial(Nx::Int, Ny::Int, l::Float64, ε::Float64,
                        vol_fraction::Float64;
                        clamp_range::Union{Nothing,Tuple{Float64,Float64}}=(0.02, 0.98))
    u  = Matrix{Float64}(undef, Nx, Ny)
    Lx = Nx * l
    Ly = Ny * l
    a  = sqrt(vol_fraction * Lx * Ly)
    xc = 0.5 * Lx
    yc = 0.5 * Ly
    for j in 1:Ny
        y = (j - 0.5) * l
        for i in 1:Nx
            x  = (i - 0.5) * l
            ux = 0.5 * (tanh((x - (xc - a/2)) / ε) - tanh((x - (xc + a/2)) / ε))
            uy = 0.5 * (tanh((y - (yc - a/2)) / ε) - tanh((y - (yc + a/2)) / ε))
            u[i, j] = ux * uy
        end
    end
    if clamp_range !== nothing
        lo, hi = clamp_range
        u .= clamp.(u, lo, hi)
    end
    return vec(u)
end

end # module
