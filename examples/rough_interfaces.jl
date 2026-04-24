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

# ╔═╡ 90918734-7fa8-bc07-be3a-e457b62941e0
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using PhaseField
    using Printf
    using Plots
    using PlutoUI
    using Random
end

# ╔═╡ 90918734-472c-4171-0061-cf6d7f8bf56a
md"# Rough Interfaces Example

This Pluto notebook solves the dimensionally-reduced phase-field capillary problem for two rough interfaces. The roughness is self-affine and generated using a Fourier-filtering algorithm.

The gap function $g(x)$ is defined as $g(x) = z + h_2(x) - h_1(x)$, where $z$ is the mean separation and $h_1, h_2$ are independent self-affine rough surfaces."

# ╔═╡ 90918734-da16-cee5-b6d0-897211bc37e7
md"## Parameters"

# ╔═╡ 90918734-428b-931a-53b5-b4722c76bba1
md"**Nx:** $(@bind Nx_slider Slider(64:64:2048, default=64, show_value=true))"

# ╔═╡ 90918734-8e34-5092-49d9-44b3f89015e2
md"**Ny:** $(@bind Ny_slider Slider(64:64:2048, default=64, show_value=true))"

# ╔═╡ 90918734-833a-549b-8ae8-2e6abc2cc640
md"**Contact angle θ (deg):** $(@bind θ_deg_slider Slider(0:5:180, default=60, show_value=true))"

# ╔═╡ 90918734-e5ee-b527-2474-12c6084d7481
md"**Interface thickness ε/l:** $(@bind ε_slider Slider(0.5:0.1:5.0, default=1.0, show_value=true))"

# ╔═╡ 90918734-016e-6426-d984-dc3b7fff9fa6
md"**Mean separation z/l:** $(@bind z_slider Slider(1:0.5:20, default=10, show_value=true))"

# ╔═╡ 90918734-4a17-a372-c5f9-ff4e59955b27
md"**Target volume fraction:** $(@bind vol_fraction_slider Slider(0.01:0.01:0.99, default=0.2, show_value=true))"

# ╔═╡ 90918734-b001-a372-c5f9-ff4e59955b27
md"### Roughness Parameters"

# ╔═╡ 90918734-b002-a372-c5f9-ff4e59955b27
md"**Hurst exponent H:** $(@bind Hurst_slider Slider(0.1:0.1:1.0, default=0.8, show_value=true))"

# ╔═╡ 90918734-b003-a372-c5f9-ff4e59955b27
md"**RMS height h_rms/l:** $(@bind h_rms_slider Slider(0.1:0.1:5.0, default=1.0, show_value=true))"

# ╔═╡ 90918734-b004-a372-c5f9-ff4e59955b27
md"**Random seed:** $(@bind seed_slider Slider(1:100, default=42, show_value=true))"

# ╔═╡ 90918734-b006-a372-c5f9-ff4e59955b27
md"**Rolloff wavelength λ\_rolloff/l (0 = no rolloff):** $(@bind rolloff_wl_slider Slider(0:4:256, default=0, show_value=true))"

# ╔═╡ 90918734-b007-a372-c5f9-ff4e59955b27
md"**Short cutoff wavelength λ\_cutoff/l (0 = Nyquist):** $(@bind cutoff_wl_slider Slider(0:2:64, default=0, show_value=true))"

# ╔═╡ 90918734-ecfb-5864-544e-78a7dd92be6b
begin
    Nx = Nx_slider
    Ny = Ny_slider
    l  = 0.01
    θ  = deg2rad(θ_deg_slider)
    σ  = -cos(θ)
    C_σ = compute_C(σ)
    ε  = ε_slider * l
    z  = z_slider * l
    vol_frac = vol_fraction_slider
    
    Hurst = Hurst_slider
    h_rms = h_rms_slider * l
    seed  = seed_slider
    long_cutoff  = rolloff_wl_slider == 0 ? nothing : rolloff_wl_slider * l
    short_cutoff = cutoff_wl_slider  == 0 ? nothing : cutoff_wl_slider  * l

    x_coords = ((1:Nx) .- 0.5) .* l
    y_coords = ((1:Ny) .- 0.5) .* l

    sx = Nx * l
    sy = Ny * l

    # Generate rough surfaces. Use a single local MersenneTwister consumed
    # sequentially — avoids mutating the global RNG and avoids the old
    # `seed, seed+1` trick that coupled consecutive seeds (bumping `seed` by 1
    # would make the new h1 equal to the old h2).
    rng = MersenneTwister(seed)
    h1 = Roughness.fourier_synthesis(Nx, Ny, sx, sy, Hurst; rms_height=h_rms,
                                     long_cutoff=long_cutoff, short_cutoff=short_cutoff, rng=rng)
    h2 = Roughness.fourier_synthesis(Nx, Ny, sx, sy, Hurst; rms_height=h_rms,
                                     long_cutoff=long_cutoff, short_cutoff=short_cutoff, rng=rng)

    gap = z .+ h2 .- h1
    # Check for overlapping regions (negative gap)
    overlap = gap .< 0
    # Physical gap: must be non-negative
    gap = max.(gap, 0.0)

    rolloff_str = rolloff_wl_slider == 0 ? "none" : "$(rolloff_wl_slider)l"
    cutoff_str  = cutoff_wl_slider  == 0 ? "Nyquist" : "$(cutoff_wl_slider)l"
    md"**Grid:** $(Nx) × $(Ny), spacing $l = $(l)$
    **Roughness:** $H = $(Hurst), h_{rms} = $(h_rms_slider)l$
    **Cutoffs:** λ\_rolloff = $(rolloff_str), λ\_cutoff = $(cutoff_str)$
    **Separation:** $z = $(z_slider)l$
    **Overlapping regions:** $(sum(overlap)) pixels"
end

# ╔═╡ 90918734-b005-a372-c5f9-ff4e59955b27
begin
    p1 = heatmap(x_coords, y_coords, h1', aspect_ratio=:equal, title="Surface 1", color=:grays)
    p2 = heatmap(x_coords, y_coords, h2', aspect_ratio=:equal, title="Surface 2", color=:grays)
    p3 = heatmap(x_coords, y_coords, gap', aspect_ratio=:equal, title="Gap g(x,y)", color=:viridis)
    p4 = heatmap(x_coords, y_coords, overlap', aspect_ratio=:equal, title="Overlap", color=:reds)
    plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 800))
end

# ╔═╡ 90918734-45fb-bd0e-3af2-4d74a2fa3d49
md"## Simulation"

# ╔═╡ 90918734-fd67-80d3-e37c-d7e7b295b2f4
begin
    # --- volume constraint ---
    u_ones = ones(Nx*Ny)
    vol_max = compute_volume(u_ones, gap, l)
    vol_target = vol_frac * vol_max
    vol_grad   = compute_volume_gradient(gap, l)

    # --- initial condition ---
    # `square_initial` clamps interior to [0.02, 0.98] by default so the solver's
    # sigmoid reparametrisation stays in its responsive band.
    u0 = square_initial(Nx, Ny, l, ε, vol_frac)
    md"Initial condition generated."
end

# ╔═╡ 90918734-996a-156f-184b-e181d9fa920d
heatmap(x_coords, y_coords, reshape(u0, Nx, Ny)', aspect_ratio=:equal, title="Initial condition u0", color=:viridis, xlabel="x", ylabel="y")

# ╔═╡ 90918734-8fb9-c879-7ac6-40a15b93b0ca
begin
    # Note: Pluto is reactive — moving any slider above re-runs the solver
    # automatically. Keep Nx, Ny modest while exploring parameters.

    energy_fn(u)      = phase_field_energy(u, gap, ε, l, σ, C_σ)
    gradient_fn!(G,u) = phase_field_gradient!(G, u, gap, ε, l, σ, C_σ)

    g_tol_val = 1e-5

    u, λ_final, residual = solve_volume_constrained_bfgs(
        energy_fn, gradient_fn!, vol_grad, u0, vol_target;
        contact = vec(overlap), g_tol = g_tol_val, verbose = false)

    V_fin = compute_volume(u, gap, l)
    E_fin = phase_field_energy(u, gap, ε, l, σ, C_σ)
    u_mat = reshape(u, Nx, Ny)

    md"Simulation complete."
end

# ╔═╡ 90918734-76cd-dce5-8932-5f3f986f1f8c
md"## Results"

# ╔═╡ 90918734-e421-25ba-aefa-db0f7a0333b1
begin
    heatmap(x_coords, y_coords, u_mat', aspect_ratio=:equal, title="Phase-field u", color=:viridis,
            xlabel="x", ylabel="y")
end

# ╔═╡ 90918734-e422-25ba-aefa-db0f7a0333b1
begin
    # Build a two-section colormap:
    #   u ∈ [0, 1]  →  full viridis  (data range maps to clims [0, 2] lower half)
    #   contact = 2.0  →  red        (clims upper half, never occupied by real data)
    viridis_g  = cgrad(:viridis)
    n_v        = 256
    pf_colors  = convert.(RGB{Float64}, [viridis_g[t] for t in range(0.0, 1.0; length=n_v)])
    push!(pf_colors, RGB(0.85, 0.15, 0.1))
    pf_cmap    = cgrad(pf_colors, vcat(collect(range(0.0, 0.5; length=n_v)), [1.0]))

    u_display           = copy(u_mat)
    u_display[overlap] .= 2.0

    heatmap(x_coords, y_coords, u_display',
            color=pf_cmap, clims=(0.0, 2.0),
            aspect_ratio=:equal, colorbar=false,
            title="Phase-field u (red = contact region)",
            xlabel="x", ylabel="y")
end

# ╔═╡ 90918734-901b-0e54-fc66-6466bfa2df0b
begin
    md"""
    ### Diagnostics
    - **Energy I_ε:** $(@sprintf("%.6e", E_fin))
    - **Volume:** $(@sprintf("%.6e", V_fin)) (target $(@sprintf("%.6e", vol_target)))
    - **Volume fraction:** $(@sprintf("%.6f", V_fin / vol_max)) (target $(vol_frac))
    - **Lagrange multiplier λ:** $(@sprintf("%.6e", λ_final)) (Laplace pressure)
    - **u range:** [$(round(minimum(u_mat), digits=4)), $(round(maximum(u_mat), digits=4))]
    - **KKT residual ‖Gₚ‖∞:** $(@sprintf("%.6e", residual)) (tolerance $(@sprintf("%.0e", g_tol_val)))
    """
end

# ╔═╡ Cell order:
# ╟─90918734-472c-4171-0061-cf6d7f8bf56a
# ╠═90918734-7fa8-bc07-be3a-e457b62941e0
# ╟─90918734-da16-cee5-b6d0-897211bc37e7
# ╠═90918734-428b-931a-53b5-b4722c76bba1
# ╠═90918734-8e34-5092-49d9-44b3f89015e2
# ╠═90918734-833a-549b-8ae8-2e6abc2cc640
# ╠═90918734-e5ee-b527-2474-12c6084d7481
# ╠═90918734-016e-6426-d984-dc3b7fff9fa6
# ╠═90918734-4a17-a372-c5f9-ff4e59955b27
# ╟─90918734-b001-a372-c5f9-ff4e59955b27
# ╠═90918734-b002-a372-c5f9-ff4e59955b27
# ╠═90918734-b003-a372-c5f9-ff4e59955b27
# ╠═90918734-b004-a372-c5f9-ff4e59955b27
# ╠═90918734-b006-a372-c5f9-ff4e59955b27
# ╠═90918734-b007-a372-c5f9-ff4e59955b27
# ╟─90918734-ecfb-5864-544e-78a7dd92be6b
# ╠═90918734-b005-a372-c5f9-ff4e59955b27
# ╟─90918734-45fb-bd0e-3af2-4d74a2fa3d49
# ╠═90918734-fd67-80d3-e37c-d7e7b295b2f4
# ╠═90918734-996a-156f-184b-e181d9fa920d
# ╠═90918734-8fb9-c879-7ac6-40a15b93b0ca
# ╟─90918734-76cd-dce5-8932-5f3f986f1f8c
# ╠═90918734-e421-25ba-aefa-db0f7a0333b1
# ╠═90918734-e422-25ba-aefa-db0f7a0333b1
# ╟─90918734-901b-0e54-fc66-6466bfa2df0b
