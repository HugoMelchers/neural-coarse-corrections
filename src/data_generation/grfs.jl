#=
Implementation of 1D and 2D GRFs with squared-exponential covariance kernels and zero mean.

1D GRF samples are written in terms of a squared-exponential kernel basis. 2D GRF samples are
written in terms of a tensor product of 1D bases.

Usage:

- `basis = GRF1D((0, 1), 0.2)` generates a GRF in 1D, supported on the interval [0, 1], and with
  correlation length 0.2. Note that the result is a `Functions` object, representing basis functions
  of the GRF.
- `samples = sample(basis, 10)` generates 10 functions sampled from the GRF. It is also possible to
  pass a specific rng
- `basis2 = GRF2D(basis)` generates a 2D GRF on [0, 1]² using a Tensor product. `GRF2D(basis_x,
  basis_y)` can be used for different limits or lengths in X and Y
- `samples2 = sample(basis2, 10)` generates 10 functions sampled from the 2D GRF.

The samples are functions, and can be called
- `samples(xs)` returns a matrix of values where each column is one function, sampled at the nodes
- `derivative(samples, xs, k)` returns the k-th derivative in the same format
- `samples2(xs, ys)` returns a 3-dimensional array with indices (x, y, b)
- `derivative(samples2, xs, (k1, k2))` differentiates k1 times in x, and k2 times in y

Written by Hugo Melchers, version 2025-03-17.
=#

using LinearAlgebra
import Base: length, getindex

struct RBF1D
    lims::NTuple{2, Float64}
    l::Float64
    nodes::Vector{Float64}
end

length(rbf::RBF1D) = length(rbf.nodes)
getindex(rbf::RBF1D, sel) = RBF1D(rbf.lims, rbf.l, rbf.nodes[sel])

_hermite(xs::AbstractArray, k) =
    if k == 0
        ones(eltype(xs), size(xs))
    elseif k == 1
        2 .* Array(xs)
    else
        Hᵢ₋₁ = ones(eltype(xs), size(xs))
        Hᵢ = 2 .* Array(xs)
        i = 1
        while i < k
            # Use the recursion Hᵢ₊₁(x) = 2x Hᵢ(x) - Hᵢ'(x) = 2x Hᵢ(x) - 2i Hᵢ₋₁(x)
            # Overwrite the array Hᵢ₋₁ and then cycle the arrays around
            @. Hᵢ₋₁ = 2xs * Hᵢ - 2i * Hᵢ₋₁
            Hᵢ₋₁, Hᵢ = Hᵢ, Hᵢ₋₁
            i += 1
        end
        Hᵢ
    end

_hermite(x::Number, k) =
    if k == 0
        one(x)
    elseif k == 1
        2x
    else
        Hᵢ₋₁ = one(x)
        Hᵢ = 2x
        i = 1
        while i < k
            # Use the recursion Hᵢ₊₁(x) = 2x Hᵢ(x) - Hᵢ'(x) = 2x Hᵢ(x) - 2i Hᵢ₋₁(x)
            # Overwrite the array Hᵢ₋₁ and then cycle the arrays around
            Hᵢ₋₁ = 2x * Hᵢ - 2i * Hᵢ₋₁
            Hᵢ₋₁, Hᵢ = Hᵢ, Hᵢ₋₁
            i += 1
        end
        Hᵢ
    end

(fs::RBF1D)(x) =
    @. exp(-abs2(x - fs.nodes') / fs.l^2)

function derivative(fs::RBF1D, x, k = 1)
    # One basis function is exp(-(x - x₀)² / l²)
    # The k-th derivative of this is exp(-(x - x₀)² / l²) * (-1)ᵏ / lᵏ * Hₖ((x - x₀) / l)
    ds = @. (x - fs.nodes') / fs.l
    basis = @. exp.(-abs2(ds)) * (-1 / fs.l)^k
    basis .*= _hermite(ds, k)
    basis
end

function GRF1D(lims, l; n_nodes = nothing, tol = 1e-14)
    a, b = lims
    L = b - a
    nr_nodes = if n_nodes === nothing
        max(ceil(Int, 4 * L / l), 30)
    else
        n_nodes
    end
    nodes = range(a, b, nr_nodes)
    basis = RBF1D(lims, l, nodes)

    K = @. exp(-abs2(nodes - nodes') / l^2)
    vals, vecs = eigen(Symmetric(K); sortby = -)
    m = findlast(>(tol * vals[1]), vals)
    vals = @view vals[1:m]
    vecs = @view vecs[:, 1:m]
    vecs ./= sqrt.(vals)'
    Functions1D(basis, Array(vecs))
end

function sample(grf::Functions1D{RBF1D}, N; rng = nothing)
    m = size(grf.weights, 2)
    Z = if rng === nothing
        randn(m, N)
    else
        randn(rng, m, N)
    end
    weights = grf.weights * Z
    Functions1D(grf.basis, weights)
end

struct GRF2D{X, Y}
    x::Functions1D{X}
    y::Functions1D{Y}
end

GRF2D(x::Functions1D) = GRF2D(x, x)

function sample(grf::GRF2D, N; rng = nothing)
    m_x = size(grf.x.weights, 2)
    m_y = size(grf.y.weights, 2)
    Z = if rng === nothing
        randn(m_x, m_y, N)
    else
        randn(rng, m_x, m_y, N)
    end
    coeffs = _mat_mats_mat(grf.x.weights, Z, grf.y.weights')
    TensorFunctions2D(grf.x.basis, grf.y.basis, coeffs)
end

flatten(funcs::RBF1D) = funcs
flatten(funcs::GRF2D) = funcs
