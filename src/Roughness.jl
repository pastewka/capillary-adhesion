module Roughness

using FFTW
using Random
using Statistics

export self_affine_prefactor, fourier_synthesis

"""
    self_affine_prefactor(nx, ny, sx, sy, Hurst; rms_height=nothing, rms_slope=nothing,
                          short_cutoff=nothing, long_cutoff=nothing, rolloff=0.0)

Calculate the prefactor for self-affine Fourier synthesis.

If `long_cutoff` is set, modes with `|q| < 2π/long_cutoff` follow a constant plateau
at spectral amplitude `rolloff · q_min^(-(1+H))` (see `fourier_synthesis`). Passing
`rolloff > 0` here includes the plateau's contribution in the variance integrals so
that `rms_height` / `rms_slope` targets are preserved. With the default `rolloff=0`
the formula assumes a pure power-law PSD on `[q_min, q_max]` (no plateau).
"""
function self_affine_prefactor(nx, ny, sx, sy, Hurst;
    rms_height=nothing,
    rms_slope=nothing,
    short_cutoff=nothing,
    long_cutoff=nothing,
    rolloff=0.0)
    if short_cutoff !== nothing
        q_max = 2π / short_cutoff
    else
        q_max = π * min(nx / sx, ny / sy)
    end

    if long_cutoff !== nothing
        q_min = 2π / long_cutoff
    else
        q_min = 2π * max(1 / sx, 1 / sy)
    end

    area = sx * sy
    r2   = rolloff^2

    # Variance contributions (up to a common FFT-normalisation constant C²):
    #   rms_height² ∝ ∫ PSD(q) d²q ∝ (q_min^(-2H)·(1 + H·rolloff²) − q_max^(-2H)) / (2H)
    #   rms_slope²  ∝ ∫ q²·PSD(q) d²q
    #               ∝ (q_max^(2−2H) − q_min^(2−2H)·(1 − (1−H)·rolloff²/2)) / (2−2H)
    # The plateau terms (the `rolloff²` corrections) vanish for rolloff=0, recovering
    # the pure power-law expressions used before.
    if rms_height !== nothing
        denom = q_min^(-2 * Hurst) * (1 + Hurst * r2) - q_max^(-2 * Hurst)
        fac = 2 * rms_height * sqrt(Hurst * π) / sqrt(denom)
    elseif rms_slope !== nothing
        denom = q_max^(2 - 2 * Hurst) - q_min^(2 - 2 * Hurst) * (1 - (1 - Hurst) * r2 / 2)
        fac = 2 * rms_slope * sqrt((1 - Hurst) * π) / sqrt(denom)
    else
        error("Neither rms height nor rms slope is defined!")
    end
    return fac * nx * ny / sqrt(area)
end

"""
    fourier_synthesis(nx, ny, sx, sy, Hurst;
                      rms_height=nothing, rms_slope=nothing,
                      short_cutoff=nothing, long_cutoff=nothing,
                      rolloff=1.0, rng=Random.default_rng())

Create a self-affine, randomly rough surface using a Fourier-filtering algorithm.

The spectral amplitude follows a power law `|q|^(-(1+H))` on `[q_min, q_max]`,
where `q_max = 2π/short_cutoff` (or Nyquist) and `q_min = 2π/long_cutoff`. When
`long_cutoff` is set, modes below `q_min` are held at a plateau equal to
`rolloff · q_min^(-(1+H))`; the prefactor is renormalised accordingly so that the
requested `rms_height` / `rms_slope` is preserved.

Pass `rng` to use a non-global RNG (e.g. an independent `MersenneTwister(seed)`
for each surface) without mutating global state.
"""
function fourier_synthesis(nx, ny, sx, sy, Hurst;
    rms_height=nothing,
    rms_slope=nothing,
    short_cutoff=nothing,
    long_cutoff=nothing,
    rolloff=1.0,
    rng::AbstractRNG=Random.default_rng())
    if short_cutoff !== nothing
        q_max = 2π / short_cutoff
    else
        q_max = π * min(nx / sx, ny / sy)
    end

    if long_cutoff !== nothing
        q_min = 2π / long_cutoff
    else
        q_min = nothing
    end

    # When no long_cutoff is given there is no plateau region, so the prefactor
    # must assume rolloff = 0 regardless of the `rolloff` keyword.
    prefactor_rolloff = long_cutoff === nothing ? 0.0 : rolloff
    fac = self_affine_prefactor(nx, ny, sx, sy, Hurst;
        rms_height=rms_height,
        rms_slope=rms_slope,
        short_cutoff=short_cutoff,
        long_cutoff=long_cutoff,
        rolloff=prefactor_rolloff)

    kny = ny ÷ 2 + 1
    karr = zeros(ComplexF64, nx, kny)

    qy = 2π .* (0:kny-1) ./ sy
    for x in 0:nx-1
        if x > nx ÷ 2
            qx = 2π * (nx - x) / sx
        else
            qx = 2π * x / sx
        end

        q_sq = qx^2 .+ qy .^ 2
        # Avoid division by zero at q=0
        q_sq_reg = copy(q_sq)
        if x == 0
            q_sq_reg[1] = 1.0
        end

        phase = exp.(2π .* rand(rng, kny) .* 1im)
        ran = fac .* phase .* randn(rng, kny)

        karr[x+1, :] = ran .* q_sq_reg .^ (-(1 + Hurst) / 2)

        # Apply q_max cutoff
        karr[x+1, q_sq.>q_max^2] .= 0.0

        # Apply q_min rolloff: modes below q_min are held at the spectral amplitude
        # evaluated at q_min, i.e. ran * q_min^(-(1+Hurst)), scaled by rolloff.
        # This is consistent with the main formula ran * q_sq^(-(1+Hurst)/2) evaluated
        # at |q| = q_min, since q_min^(2*(-(1+Hurst)/2)) = q_min^(-(1+Hurst)).
        if q_min !== nothing
            mask = q_sq .< q_min^2
            karr[x+1, mask] .= rolloff .* ran[mask] .* q_min^(-(1 + Hurst))
        end
    end

    # Enforce Hermitian symmetry for real output
    # H(qx, qy) = H(-qx, -qy)*. In half-spectrum, only qy=0 and qy=ny/2 (if exists)
    # have constraints on the qx components because they are their own conjugates.
    # qy=0 is the first column, qy=ny/2 is the (ny/2+1)-th column.

    # Enforce it for qy=0
    # The DC component (q=0) must be zero to give a zero-mean surface.
    karr[1, 1] = 0.0
    if nx % 2 == 0
        for x in 1:(nx÷2-1)
            karr[nx-x+1, 1] = conj(karr[x+1, 1])
        end
        karr[nx÷2+1, 1] = real(karr[nx÷2+1, 1])
    else
        for x in 1:(nx÷2)
            karr[nx-x+1, 1] = conj(karr[x+1, 1])
        end
    end

    # Enforce it for qy=ny/2 (if ny is even)
    if ny % 2 == 0
        kny = ny ÷ 2 + 1
        karr[1, kny] = real(karr[1, kny])
        if nx % 2 == 0
            for x in 1:(nx÷2-1)
                karr[nx-x+1, kny] = conj(karr[x+1, kny])
            end
            karr[nx÷2+1, kny] = real(karr[nx÷2+1, kny])
        else
            for x in 1:(nx÷2)
                karr[nx-x+1, kny] = conj(karr[x+1, kny])
            end
        end
    end

    # Inverse FFT
    # First dimension (nx) is full complex-to-complex
    # Second dimension (ny) is real-to-complex (so we use irfft)

    # Python code does:
    # for i in range(ncolumns): karr[:, i] = np.fft.ifft(karr[:, i])
    # for i in range(nrows): rarr[i, :] = np.fft.irfft(karr[i, :])

    # In Julia:
    # karr is (nx, kny)
    # bfft!(karr, 1) # Unnormalized inverse FFT on columns
    # rarr = irfft(karr, ny, 2)

    # Note: np.fft.ifft is normalized by 1/n. Julia's ifft is too.
    # Julia's irfft is also normalized.

    karr_ifft = ifft(karr, 1)
    rarr = irfft(karr_ifft, ny, 2)

    return rarr
end

end # module
