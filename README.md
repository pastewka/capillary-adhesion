# PhaseField.jl

A Julia package for solving dimensionally-reduced phase-field capillary problems.

## Installation

To install this package, use the Julia package manager:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Examples

### Parallel Plates

The `examples/parallel_plates.jl` Pluto notebook solves the dimensionally-reduced phase-field capillary problem for two flat, parallel plates with an adjustable volume fraction.

### Rough Interfaces

The `examples/rough_interfaces.jl` Pluto notebook solves the dimensionally-reduced phase-field capillary problem for two rough, self-affine interfaces. The roughness is generated using a Fourier-filtering algorithm.

To run either notebook:

```bash
cd examples
julia --project -e 'using Pluto; Pluto.run(notebook="parallel_plates.jl")' # or rough_interfaces.jl
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
