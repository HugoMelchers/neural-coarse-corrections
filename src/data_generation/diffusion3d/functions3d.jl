#=
3D equivalents of the tensor-product functions in `functions2d.jl`.

A 3D function is written as f(x, y, z) = Σ_{abc} C[a, b, c] φᵃ(x) ψᵇ(y) χᶜ(z), where φ, ψ, χ are 1D
bases (`BSpline`, `Functions1D`, ...). Batches of functions store their coefficients in a 4D array
with indices [a, b, c, sample].

Evaluating on a tensor grid is done mode-by-mode (first contract over a, then b, then c), which is
much cheaper than contracting over all three indices at once.

Requires `functions1d.jl` and `functions2d.jl` to be loaded (for `⊗`, `Functions1D` and `flatten`).
=#

import Base: length, getindex, firstindex, lastindex, view
import LinearAlgebra: I, UniformScaling
import SparseArrays: AbstractSparseArray
using Tullio

# 3D tensor grid
struct TensorGrid3D{X <: AbstractVector, Y <: AbstractVector, Z <: AbstractVector}
    xs::X
    ys::Y
    zs::Z
end

# Allows writing `xs ⊗ ys ⊗ zs`, which parses as `(xs ⊗ ys) ⊗ zs`
⊗(grid::TensorGrid2D, zs::AbstractVector) = TensorGrid3D(grid.xs, grid.ys, zs)

function tofull(grid::TensorGrid3D)
    nx, ny, nz = length(grid.xs), length(grid.ys), length(grid.zs)
    hcat(
        repeat(grid.xs; outer = ny * nz),
        repeat(grid.ys; inner = nx, outer = nz),
        repeat(grid.zs; inner = nx * ny),
    )
end

# Tullio indexes element-by-element, which is very slow for sparse matrices
_dense(A::AbstractSparseArray) = Array(A)
_dense(A) = A

#=
Mode products: Out[i, j, k, s] = Σ_{abc} Ax[i, a] Ay[j, b] Az[k, c] C[a, b, c, s]

Any of the matrices may be a `UniformScaling` (e.g. `I`), in which case that mode is only scaled.
=#
_mode_x(A::UniformScaling, C) = A.λ .* C
_mode_y(A::UniformScaling, C) = A.λ .* C
_mode_z(A::UniformScaling, C) = A.λ .* C
_mode_x(A, C) = (A = _dense(A); @tullio out[i, b, c, s] := A[i, a] * C[a, b, c, s])
_mode_y(A, C) = (A = _dense(A); @tullio out[i, j, c, s] := A[j, b] * C[i, b, c, s])
_mode_z(A, C) = (A = _dense(A); @tullio out[i, j, k, s] := A[k, c] * C[i, j, c, s])

_mode_products(Ax, Ay, Az, C::AbstractArray{<:Any, 4}) = _mode_z(Az, _mode_y(Ay, _mode_x(Ax, C)))

function _mode_products(Ax, Ay, Az, C::AbstractArray{<:Any, 3})
    out = _mode_products(Ax, Ay, Az, reshape(C, size(C)..., 1))
    reshape(out, size(out, 1), size(out, 2), size(out, 3))
end

# Single function in 3D
struct TensorFunction3D{X, Y, Z, TF <: AbstractArray}
    x::X
    y::Y
    z::Z
    coefficients::TF

    function TensorFunction3D(x::X, y::Y, z::Z, coefficients::TF) where {X, Y, Z, TF}
        @assert ndims(coefficients) == 3
        @assert length(x) == size(coefficients, 1)
        @assert length(y) == size(coefficients, 2)
        @assert length(z) == size(coefficients, 3)
        new{X, Y, Z, TF}(x, y, z, coefficients)
    end
end

function derivative(fun::TensorFunction3D, grid::TensorGrid3D, k)
    dx, dy, dz = k
    Vx = derivative(fun.x, grid.xs, dx)
    Vy = derivative(fun.y, grid.ys, dy)
    Vz = derivative(fun.z, grid.zs, dz)
    _mode_products(Vx, Vy, Vz, fun.coefficients)
end

function derivative(fun::TensorFunction3D, X::AbstractArray, Y::AbstractArray, Z::AbstractArray, k)
    dx, dy, dz = k
    Vx = _dense(derivative(fun.x, view(X, :), dx))
    Vy = _dense(derivative(fun.y, view(Y, :), dy))
    Vz = _dense(derivative(fun.z, view(Z, :), dz))
    C = fun.coefficients
    @tullio out[n] := Vx[n, a] * Vy[n, b] * Vz[n, c] * C[a, b, c]
    reshape(out, size(X))
end

(fun::TensorFunction3D)(grid::TensorGrid3D) = derivative(fun, grid, (0, 0, 0))
(fun::TensorFunction3D)(X::AbstractArray, Y::AbstractArray, Z::AbstractArray) =
    derivative(fun, X, Y, Z, (0, 0, 0))

# Multiple functions in 3D
struct TensorFunctions3D{X, Y, Z, TF <: AbstractArray}
    x::X
    y::Y
    z::Z
    coefficients::TF

    function TensorFunctions3D(x::X, y::Y, z::Z, coefficients::TF) where {X, Y, Z, TF}
        @assert ndims(coefficients) == 4
        @assert length(x) == size(coefficients, 1)
        @assert length(y) == size(coefficients, 2)
        @assert length(z) == size(coefficients, 3)
        new{X, Y, Z, TF}(x, y, z, coefficients)
    end
end

function derivative(funs::TensorFunctions3D, grid::TensorGrid3D, k)
    dx, dy, dz = k
    Vx = derivative(funs.x, grid.xs, dx)
    Vy = derivative(funs.y, grid.ys, dy)
    Vz = derivative(funs.z, grid.zs, dz)
    _mode_products(Vx, Vy, Vz, funs.coefficients)
end

function derivative(
    funs::TensorFunctions3D,
    X::AbstractArray,
    Y::AbstractArray,
    Z::AbstractArray,
    k,
)
    dx, dy, dz = k
    Vx = _dense(derivative(funs.x, view(X, :), dx))
    Vy = _dense(derivative(funs.y, view(Y, :), dy))
    Vz = _dense(derivative(funs.z, view(Z, :), dz))
    C = funs.coefficients
    @tullio out[n, s] := Vx[n, a] * Vy[n, b] * Vz[n, c] * C[a, b, c, s]
    reshape(out, size(X)..., length(funs))
end

(funs::TensorFunctions3D)(grid::TensorGrid3D) = derivative(funs, grid, (0, 0, 0))
(funs::TensorFunctions3D)(X::AbstractArray, Y::AbstractArray, Z::AbstractArray) =
    derivative(funs, X, Y, Z, (0, 0, 0))

firstindex(funs::TensorFunctions3D) = 1
lastindex(funs::TensorFunctions3D) = size(funs.coefficients, 4)
length(funs::TensorFunctions3D) = size(funs.coefficients, 4)
function getindex(funs::TensorFunctions3D, idx::Integer)
    TensorFunction3D(funs.x, funs.y, funs.z, funs.coefficients[:, :, :, idx])
end
function getindex(funs::TensorFunctions3D, ids::AbstractVector)
    TensorFunctions3D(funs.x, funs.y, funs.z, funs.coefficients[:, :, :, ids])
end
function view(funs::TensorFunctions3D, idx::Integer)
    TensorFunction3D(funs.x, funs.y, funs.z, view(funs.coefficients, :, :, :, idx))
end
function view(funs::TensorFunctions3D, ids::AbstractVector)
    TensorFunctions3D(funs.x, funs.y, funs.z, view(funs.coefficients, :, :, :, ids))
end

function flatten(funcs::TensorFunctions3D)
    # If any of the 1D bases is itself a linear combination of functions, express the coefficients
    # in terms of the underlying basis directly
    fx = flatten(funcs.x)
    fy = flatten(funcs.y)
    fz = flatten(funcs.z)
    Ax, Ay, Az = I, I, I
    if fx isa Functions1D
        Ax = fx.coefficients
        fx = fx.basis
    end
    if fy isa Functions1D
        Ay = fy.coefficients
        fy = fy.basis
    end
    if fz isa Functions1D
        Az = fz.coefficients
        fz = fz.basis
    end
    coeffs = _mode_products(Ax, Ay, Az, funcs.coefficients)
    return TensorFunctions3D(fx, fy, fz, coeffs)
end
