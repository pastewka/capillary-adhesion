"""
examples/parallel_plates.jl

Solve the dimensionally-reduced phase-field capillary problem (paper eq. I_ε)
for two flat, parallel plates at 50% volume fraction.

Physical setup
──────────────
Two infinite flat plates at separation z = 10l form a uniform gap g(x) = z.
The equilibrium liquid shape that minimises the weighted perimeter at 50%
filling is a stripe (planar liquid–vapour interface crossing the periodic domain).

Numerical method
────────────────
Uses `solve_volume_constrained` from the PhaseField library:

  - Augmented-Lagrangian outer loop with box-constrained L-BFGS as the inner solver
  - Volume constraint: V[u] = vol_target, enforced via dual ascent
  - Box constraint 0 ≤ u ≤ 1: prevents unphysical values driven by the wetting term

Run from the project root:
  julia --project examples/parallel_plates.jl
"""

using PhaseField
using Printf

# ─── grid ─────────────────────────────────────────────────────────────────────
const Nx, Ny = 64, 64          # increase to 1024×1024 for paper-quality results
const l      = 0.01            # grid spacing

# ─── physics ──────────────────────────────────────────────────────────────────
const θ      = π / 3           # contact angle: 60° (hydrophilic, σ < 0)
const σ      = -cos(θ)         # Young–Dupré: σ = −cos θ ≈ −0.5
const C_σ    = compute_C(σ)    # capillary integral C(σ) ≈ 0.957
const ε      = 1.0 * l         # diffuse-interface thickness (minimum: 1 grid spacing)

# ─── geometry: flat parallel plates ──────────────────────────────────────────
const z      = 10.0 * l        # uniform plate separation
const gap    = fill(z, Nx, Ny)

# ─── volume constraint ────────────────────────────────────────────────────────
const vol_max    = z * Nx * Ny * l^2   # total available volume (gap fully filled)
const vol_target = 0.5 * vol_max       # 50% filling
const vol_grad   = vec(gap) .* l^2     # ∂V/∂u[i,j] = g[i,j]·l²

# ─── initial condition ────────────────────────────────────────────────────────
# Stripe: u = 1 (liquid) on the left half, u = 0 (vapour) on the right half,
# connected by a tanh profile of width ε.  Satisfies the volume constraint
# exactly by symmetry.
function stripe_initial(Nx, Ny, l, ε)
    u = Matrix{Float64}(undef, Nx, Ny)
    x_mid = 0.5 * Nx * l
    for j in 1:Ny, i in 1:Nx
        x       = (i - 0.5) * l
        u[i, j] = 0.5 * (1.0 + tanh((x_mid - x) / ε))
    end
    return vec(u)
end

# ─── run ──────────────────────────────────────────────────────────────────────
function main()
    println("Flat parallel plates")
    @printf("  Grid:    %d × %d,  spacing l = %.4f\n", Nx, Ny, l)
    @printf("  Gap:     z = %.1fl\n", z / l)
    @printf("  Physics: θ = %d°,  σ = %.4f,  C(σ) = %.4f\n",
            round(Int, rad2deg(θ)), σ, C_σ)
    @printf("  ε = %.1fl  (interface thickness)\n", ε / l)
    @printf("  V* = %.4e  (50%% filling)\n\n", vol_target)

    u0 = stripe_initial(Nx, Ny, l, ε)

    energy_fn(u)      = phase_field_energy(u, gap, ε, l, σ, C_σ)
    gradient_fn!(G,u) = phase_field_gradient!(G, u, gap, ε, l, σ, C_σ)
    volume_fn(u)      = compute_volume(u, gap, l)

    u, λ_final = solve_volume_constrained(
        energy_fn, gradient_fn!, volume_fn, vol_grad, u0, vol_target;
        tol_h = 1e-7, verbose = true)

    # ─── diagnostics ──────────────────────────────────────────────────────────
    V_fin = compute_volume(u, gap, l)
    E_fin = phase_field_energy(u, gap, ε, l, σ, C_σ)
    u_mat = reshape(u, Nx, Ny)

    println()
    println("─── final state ─────────────────────────────────────────")
    @printf("  Energy I_ε:          %+.6e\n", E_fin)
    @printf("  Volume:               %.6e  (target %.6e)\n", V_fin, vol_target)
    @printf("  Volume fraction:      %.6f  (target 0.5)\n", V_fin / vol_max)
    @printf("  Lagrange mult. λ:    %+.6e  (Laplace pressure)\n", λ_final)
    @printf("  u ∈ [%.4f, %.4f]  (sharp: near 0 and 1)\n", minimum(u_mat), maximum(u_mat))

    # Cross-section along x at the mid-row — confirms stripe geometry
    println("\nPhase-field profile u(x, y=L/2):")
    @printf("  %8s   %s\n", "x / l", "u")
    jmid = Ny ÷ 2
    step = max(1, Nx ÷ 16)
    for i in 1:step:Nx
        @printf("  %8.1f   %.4f\n", (i - 0.5), u_mat[i, jmid])
    end
end

main()
