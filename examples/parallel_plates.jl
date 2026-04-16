### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ c99e8e84-7fa8-bc07-be3a-e457b62941e0
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using PhaseField
    using Printf
    using Plots
    using PlutoUI
end

# ╔═╡ 1d194b7f-472c-4171-0061-cf6d7f8bf56a
md"# Parallel Plates Example

This Pluto notebook solves the dimensionally-reduced phase-field capillary problem for two flat, parallel plates with adjustable volume fraction.

The energy functional is given by (paper eq. $I_\varepsilon$):
$$I_\varepsilon[u] = \int_\omega \left\{ g(x) C(\sigma) \frac{1}{\beta[W]} \left( \varepsilon |\nabla u|^2 + \frac{W(u)}{\varepsilon} \right) + 2\sigma u \right\} dx$$

The physical setup consists of two infinite flat plates at separation $z$ forming a uniform gap $g(x) = z$. The equilibrium liquid shape that minimises the weighted perimeter is a stripe or a droplet depending on the volume fraction and contact angle. We start with a square droplet in the middle of the domain."

# ╔═╡ da1607a3-4d39-cee5-b6d0-897211bc37e7
md"## Parameters

All physical lengths in this simulation are expressed in terms of the grid spacing $l = 0.01$, which acts as the fundamental unit of length."

# ╔═╡ 428b931a-53b5-b472-2c76-bba1b968cb97
md"**Nx:** $(@bind Nx_slider Slider(64:64:1024, default=256, show_value=true))"

# ╔═╡ 8e345092-49d9-44b3-f890-15e23821d0e2
md"**Ny:** $(@bind Ny_slider Slider(64:64:1024, default=256, show_value=true))"

# ╔═╡ 833a549b-8ae8-2e6a-bc2c-c640e0c30006
md"**Contact angle θ (deg):** $(@bind θ_deg_slider Slider(0:5:180, default=60, show_value=true))"

# ╔═╡ e5eeb527-2474-12c6-084d-7481a2f72697
md"**Interface thickness ε/l:** $(@bind ε_slider Slider(0.5:0.1:5.0, default=1.0, show_value=true))"

# ╔═╡ 016e6426-d984-dc3b-7fff-9fa6ae445d46
md"**Plate separation z/l:** $(@bind z_slider Slider(1:0.5:20, default=10, show_value=true))"

# ╔═╡ 4a17a372-c5f9-ff4e-5995-5b2745e875da
md"**Target volume fraction:** $(@bind vol_fraction_slider Slider(0.01:0.01:0.99, default=0.2, show_value=true))"

# ╔═╡ ecfb5864-544e-78a7-dd92-be6bf02ecb05
begin
    Nx = Nx_slider
    Ny = Ny_slider
    l  = 0.01  # grid spacing; the characteristic length scale for the discretization
    θ  = deg2rad(θ_deg_slider)
    σ  = -cos(θ)
    C_σ = compute_C(σ)
    ε  = ε_slider * l  # interface thickness (multiple of l)
    z  = z_slider * l  # plate separation (multiple of l)
    vol_frac = vol_fraction_slider
    
    x_coords = ((1:Nx) .- 0.5) .* l
    y_coords = ((1:Ny) .- 0.5) .* l
    
    md"**Grid:** $(Nx) × $(Ny), spacing $l = $(l)$
    **Physics:** $θ = $(θ_deg_slider)°, σ = $(round(σ, digits=4)), C(σ) = $(round(C_σ, digits=4))$
    **ε:** $(ε_slider)l$ (interface thickness)
    **Separation:** $z = $(z_slider)l$ (expressed as a multiple of the grid spacing $l$)
    **Target volume fraction:** $(vol_frac)"
end

# ╔═╡ 45fbbd0e-3af2-4d74-a2fa-3d49ec88a3fe
md"## Simulation"

# ╔═╡ fd6780d3-e37c-d7e7-b295-b2f4130c02a9
begin
    # --- volume constraint ---
    vol_max    = z * Nx * Ny * l^2
    vol_target = vol_fraction_slider * vol_max
    gap        = fill(z, Nx, Ny)
    vol_grad   = vec(gap) .* l^2

    # --- initial condition ---
    function square_initial(Nx, Ny, l, ε, vol_fraction)
        u = Matrix{Float64}(undef, Nx, Ny)
        Lx = Nx * l
        Ly = Ny * l
        # Area = vol_fraction * Lx * Ly
        a = sqrt(vol_fraction * Lx * Ly)
        xc = 0.5 * Lx
        yc = 0.5 * Ly
        for j in 1:Ny
            y = (j - 0.5) * l
            for i in 1:Nx
                x = (i - 0.5) * l
                # Smooth square: 1 inside, 0 outside
                ux = 0.5 * (tanh((x - (xc - a/2)) / ε) - tanh((x - (xc + a/2)) / ε))
                uy = 0.5 * (tanh((y - (yc - a/2)) / ε) - tanh((y - (yc + a/2)) / ε))
                u[i, j] = ux * uy
            end
        end
        return vec(u)
    end

    u0 = square_initial(Nx, Ny, l, ε, vol_fraction_slider)
    md"Initial condition (square droplet) generated."
end

# ╔═╡ 996a156f-184b-e181-d9fa-920d6d218f98
heatmap(x_coords, y_coords, reshape(u0, Nx, Ny)', aspect_ratio=:equal, title="Initial condition u0", color=:viridis, xlabel="x", ylabel="y")

# ╔═╡ cb4233bf-a9fb-ad9a-cce6-9163922ee12e
@bind run_button Button("Run Simulation")

# ╔═╡ 8fb9c879-7ac6-40a1-5b93-b0ca50ad3e81
begin
    run_button
    
    energy_fn(u)      = phase_field_energy(u, gap, ε, l, σ, C_σ)
    gradient_fn!(G,u) = phase_field_gradient!(G, u, gap, ε, l, σ, C_σ)
    volume_fn(u)      = compute_volume(u, gap, l)

    u, λ_final = solve_volume_constrained(
        energy_fn, gradient_fn!, volume_fn, vol_grad, u0, vol_target;
        tol_h = 1e-7, verbose = false)

    V_fin = compute_volume(u, gap, l)
    E_fin = phase_field_energy(u, gap, ε, l, σ, C_σ)
    u_mat = reshape(u, Nx, Ny)
    
    md"Simulation complete."
end

# ╔═╡ 76cddce5-8932-5f3f-986f-1f8c2aa0c599
md"## Results"

# ╔═╡ e42125ba-aefa-db0f-7a03-33b13b846840
begin
    heatmap(x_coords, y_coords, u_mat', aspect_ratio=:equal, title="Phase-field u", color=:viridis,
            xlabel="x", ylabel="y")
end

# ╔═╡ b2701ae8-6b3a-0f44-a5a5-72b6983a5fda
begin
    jmid = Ny ÷ 2
    plot(x_coords, u_mat[:, jmid], xlabel="x", ylabel="u",
         title="Phase-field profile u(x, y=L/2)", label="u(x)")
end

# ╔═╡ 901b0e54-fc66-6466-bfa2-df0b73b89b34
begin
    md"""
    ### Diagnostics
    - **Energy I_ε:** $(@sprintf("%.6e", E_fin))
    - **Volume:** $(@sprintf("%.6e", V_fin)) (target $(@sprintf("%.6e", vol_target)))
    - **Volume fraction:** $(@sprintf("%.6f", V_fin / vol_max)) (target $(vol_fraction_slider))
    - **Lagrange multiplier λ:** $(@sprintf("%.6e", λ_final)) (Laplace pressure)
    - **u range:** [$(round(minimum(u_mat), digits=4)), $(round(maximum(u_mat), digits=4))]
    """
end

# ╔═╡ Cell order:
# ╟─1d194b7f-472c-4171-0061-cf6d7f8bf56a
# ╠═c99e8e84-7fa8-bc07-be3a-e457b62941e0
# ╟─da1607a3-4d39-cee5-b6d0-897211bc37e7
# ╠═428b931a-53b5-b472-2c76-bba1b968cb97
# ╠═8e345092-49d9-44b3-f890-15e23821d0e2
# ╠═833a549b-8ae8-2e6a-bc2c-c640e0c30006
# ╠═e5eeb527-2474-12c6-084d-7481a2f72697
# ╠═016e6426-d984-dc3b-7fff-9fa6ae445d46
# ╠═4a17a372-c5f9-ff4e-5995-5b2745e875da
# ╟─ecfb5864-544e-78a7-dd92-be6bf02ecb05
# ╟─45fbbd0e-3af2-4d74-a2fa-3d49ec88a3fe
# ╠═fd6780d3-e37c-d7e7-b295-b2f4130c02a9
# ╠═996a156f-184b-e181-d9fa-920d6d218f98
# ╠═cb4233bf-a9fb-ad9a-cce6-9163922ee12e
# ╠═8fb9c879-7ac6-40a1-5b93-b0ca50ad3e81
# ╟─76cddce5-8932-5f3f-986f-1f8c2aa0c599
# ╠═e42125ba-aefa-db0f-7a03-33b13b846840
# ╠═b2701ae8-6b3a-0f44-a5a5-72b6983a5fda
# ╟─901b0e54-fc66-6466-bfa2-df0b73b89b34
