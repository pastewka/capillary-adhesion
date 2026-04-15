using PhaseField
using Test
using Random
using Optim

@testset "PhaseField Model Tests" begin

    @testset "compute_C" begin
        # For σ = 0, C(0) = 1.0
        @test isapprox(compute_C(0.0), 1.0, atol=1e-6)

        # C(σ) = √(1-σ²)/2 + arcsin(σ)/(2σ); check σ = 0.5
        val = 0.5 * sqrt(1.0 - 0.5^2) + asin(0.5) / (2.0 * 0.5)
        @test isapprox(compute_C(0.5), val, atol=1e-6)

        # Continuity at σ = 0: Taylor and exact must agree closely
        @test isapprox(compute_C(1e-7), compute_C(0.0), rtol=1e-6)
    end

    @testset "generate_topography" begin
        Nx, Ny = 32, 32
        l = 0.01
        h = generate_topography(Nx, Ny, l; amplitude=0.15*l, seed=42)
        @test size(h) == (Nx, Ny)
        @test maximum(h) <= 0.15 * l
        @test minimum(h) >= -0.15 * l
    end

    @testset "compute_volume" begin
        Nx, Ny = 16, 16
        l = 0.1
        u = ones(Float64, Nx, Ny)
        g = ones(Float64, Nx, Ny) .* 2.0

        # Volume = sum(1.0 * 2.0 * l^2) = 16 * 16 * 2.0 * 0.01 = 5.12
        vol = compute_volume(vec(u), g, l)
        @test isapprox(vol, Nx * Ny * 2.0 * l^2, atol=1e-6)
    end

    @testset "Energy and Gradient consistency — full gradient, uniform g" begin
        # Check that the analytical gradient matches finite differences at every node
        Nx, Ny = 8, 8
        l = 0.1
        ε = 0.02
        σ_val = -0.5
        C_σ = compute_C(σ_val)

        Random.seed!(123)
        u_vec = rand(Float64, Nx * Ny)
        g = ones(Float64, Nx, Ny) .* 1.5

        G_vec = zeros(Float64, Nx * Ny)
        phase_field_gradient!(G_vec, u_vec, g, ε, l, σ_val, C_σ)

        delta = 1e-6
        for k in 1:Nx*Ny
            u_p = copy(u_vec); u_p[k] += delta
            u_m = copy(u_vec); u_m[k] -= delta
            fd = (phase_field_energy(u_p, g, ε, l, σ_val, C_σ) -
                  phase_field_energy(u_m, g, ε, l, σ_val, C_σ)) / (2.0 * delta)
            @test isapprox(G_vec[k], fd, rtol=1e-4, atol=1e-8)
        end
    end

    @testset "Energy and Gradient consistency — spatially varying g" begin
        # With a non-uniform gap the FEM interpolation of g matters; verify gradient still exact
        Nx, Ny = 6, 6
        l = 0.1
        ε = 0.05
        σ_val = 0.3
        C_σ = compute_C(σ_val)

        Random.seed!(42)
        u_vec = rand(Float64, Nx * Ny)
        g = 1.0 .+ rand(Float64, Nx, Ny) .* 0.5   # spatially varying gap in [1, 1.5]

        G_vec = zeros(Float64, Nx * Ny)
        phase_field_gradient!(G_vec, u_vec, g, ε, l, σ_val, C_σ)

        delta = 1e-6
        for k in 1:Nx*Ny
            u_p = copy(u_vec); u_p[k] += delta
            u_m = copy(u_vec); u_m[k] -= delta
            fd = (phase_field_energy(u_p, g, ε, l, σ_val, C_σ) -
                  phase_field_energy(u_m, g, ε, l, σ_val, C_σ)) / (2.0 * delta)
            @test isapprox(G_vec[k], fd, rtol=1e-4, atol=1e-8)
        end
    end

    @testset "Energy vanishes for constant u at a minimiser of W" begin
        # u ≡ 0 (or u ≡ 1) has zero gradient energy and W(u)=0; the wetting term 2σu
        # is linear so the total energy is just 2σ * u * |ω|
        Nx, Ny = 4, 4
        l = 0.1
        ε = 0.01
        σ_val = -0.5
        C_σ = compute_C(σ_val)
        g = ones(Float64, Nx, Ny)
        area = Nx * Ny * l^2

        u0 = zeros(Float64, Nx * Ny)
        @test isapprox(phase_field_energy(u0, g, ε, l, σ_val, C_σ), 0.0, atol=1e-12)

        u1 = ones(Float64, Nx * Ny)
        @test isapprox(phase_field_energy(u1, g, ε, l, σ_val, C_σ), 2.0 * σ_val * area, rtol=1e-10)
    end

    @testset "Periodicity: energy is translation-invariant" begin
        # Shifting u by one cell along x must leave the periodic energy unchanged
        Nx, Ny = 8, 8
        l = 0.1
        ε = 0.03
        σ_val = 0.0
        C_σ = compute_C(σ_val)
        g = ones(Float64, Nx, Ny)

        Random.seed!(7)
        u = rand(Float64, Nx, Ny)
        E0 = phase_field_energy(vec(u), g, ε, l, σ_val, C_σ)

        u_shifted = circshift(u, (1, 0))
        E1 = phase_field_energy(vec(u_shifted), g, ε, l, σ_val, C_σ)
        @test isapprox(E0, E1, rtol=1e-12)
    end

    @testset "solve_volume_constrained — flat gap (parallel plates)" begin
        # Small grid (32×32) with uniform gap; solve at 50% filling.
        # Expect convergence and u ∈ [0,1].
        Nx, Ny = 32, 32
        l      = 0.01
        ε      = 1.0 * l
        σ_val  = -cos(π / 3)   # 60° contact angle
        C_σ    = compute_C(σ_val)
        z      = 10.0 * l
        g      = fill(z, Nx, Ny)

        vol_max    = z * Nx * Ny * l^2
        vol_target = 0.5 * vol_max
        vol_grad   = vec(g) .* l^2

        # Stripe initial condition (satisfies volume constraint by symmetry)
        u0 = let
            u_tmp = Matrix{Float64}(undef, Nx, Ny)
            x_mid = 0.5 * Nx * l
            for j in 1:Ny, i in 1:Nx
                x = (i - 0.5) * l
                u_tmp[i, j] = 0.5 * (1.0 + tanh((x_mid - x) / ε))
            end
            vec(u_tmp)
        end

        energy_fn(u)      = phase_field_energy(u, g, ε, l, σ_val, C_σ)
        gradient_fn!(G,u) = phase_field_gradient!(G, u, g, ε, l, σ_val, C_σ)
        volume_fn(u)      = compute_volume(u, g, l)

        u_sol, λ_sol = solve_volume_constrained(
            energy_fn, gradient_fn!, volume_fn, vol_grad, u0, vol_target;
            tol_h = 1e-6, verbose = false)

        V_sol = compute_volume(u_sol, g, l)
        @test isapprox(V_sol, vol_target, atol=1e-5)   # volume constraint satisfied
        @test all(u_sol .>= 0.0)                        # lower bound respected
        @test all(u_sol .<= 1.0)                        # upper bound respected
        @test isfinite(λ_sol)                           # Lagrange multiplier finite
    end

    @testset "solve_volume_constrained — spatially varying gap" begin
        # Non-uniform gap to exercise the generalisation; check volume constraint only.
        Nx, Ny = 16, 16
        l      = 0.01
        ε      = 1.0 * l
        σ_val  = -cos(π / 4)   # 45° contact angle
        C_σ    = compute_C(σ_val)

        Random.seed!(99)
        z_mean = 10.0 * l
        # Rough gap: mean z_mean with ±20% variation, always positive
        g = z_mean .* (1.0 .+ 0.2 .* (rand(Float64, Nx, Ny) .- 0.5) .* 2)
        g = max.(g, 0.01 * l)   # ensure positive gap everywhere

        vol_max    = sum(g) * l^2
        vol_target = 0.5 * vol_max
        vol_grad   = vec(g) .* l^2

        u0 = let
            u_tmp = Matrix{Float64}(undef, Nx, Ny)
            x_mid = 0.5 * Nx * l
            for j in 1:Ny, i in 1:Nx
                x = (i - 0.5) * l
                u_tmp[i, j] = 0.5 * (1.0 + tanh((x_mid - x) / ε))
            end
            vec(u_tmp)
        end

        energy_fn(u)      = phase_field_energy(u, g, ε, l, σ_val, C_σ)
        gradient_fn!(G,u) = phase_field_gradient!(G, u, g, ε, l, σ_val, C_σ)
        volume_fn(u)      = compute_volume(u, g, l)

        u_sol, λ_sol = solve_volume_constrained(
            energy_fn, gradient_fn!, volume_fn, vol_grad, u0, vol_target;
            tol_h = 1e-5, verbose = false)

        V_sol = compute_volume(u_sol, g, l)
        @test isapprox(V_sol, vol_target, atol=1e-4)
        @test all(u_sol .>= 0.0)
        @test all(u_sol .<= 1.0)
        @test isfinite(λ_sol)
    end

end
