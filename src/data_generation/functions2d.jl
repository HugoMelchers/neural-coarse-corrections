import Base: length, getindex, firstindex, lastindex, getindex, view

# 2D tensor grid
struct TensorGrid2D{X <: AbstractVector, Y <: AbstractVector}
    xs::X
    ys::Y
end

⊗(xs::AbstractVector, ys::AbstractVector) = TensorGrid2D(xs, ys)

function tofull(grid::TensorGrid2D)
    nx = length(grid.xs)
    ny = length(grid.ys)
    hcat(
        repeat(grid.xs; outer = (1, ny)),
        repeat(grid.ys'; outer = (nx, 1)),
    )
end

# Single function in 2D
struct TensorFunction2D{X, Y, TF <: AbstractMatrix}
    x::X
    y::Y
    coefficients::TF

    function TensorFunction2D(x::X, y::Y, coefficients::TF) where {X, Y, TF}
        @assert length(x) == size(coefficients, 1)
        @assert length(y) == size(coefficients, 2)
        new{X, Y, TF}(x, y, coefficients)
    end
end

function derivative(fun::TensorFunction2D, grid::TensorGrid2D, k)
    dx, dy = k
    Vx = derivative(fun.x, grid.xs, dx)
    Vy = derivative(fun.y, grid.ys, dy)
    Vx * fun.coefficients * Vy'
end

function derivative(fun::TensorFunction2D, X::Array, Y::Array, k)
    dx, dy = k
    N = length(X)
    Bx, By = length(fun.x), length(fun.y)
    Vx = derivative(fun.x, X, dx)
    Vy = derivative(fun.y, Y, dy)
    Vx = reshape(Vx, N, Bx, 1)
    Vy = reshape(Vy, N, 1, By)
    W = reshape(fun.coefficients, 1, Bx, By)
    reshape(sum(Vx .* Vy .* W; dims = (2, 3)), N)
end

(fun::TensorFunction2D)(grid::TensorGrid2D) = derivative(fun, grid, (0, 0))
(fun::TensorFunction2D)(X::AbstractArray, Y::AbstractArray) = derivative(fun, X, Y, (0, 0))

# Multiple functions in 2D
struct TensorFunctions2D{X, Y, TF <: AbstractArray}
    x::X
    y::Y
    coefficients::TF

    function TensorFunctions2D(x::X, y::Y, coefficients::TF) where {X, Y, TF}
        @assert ndims(coefficients) == 3
        @assert length(x) == size(coefficients, 1)
        @assert length(y) == size(coefficients, 2)
        new{X, Y, TF}(x, y, coefficients)
    end
end

function derivative(funs::TensorFunctions2D, grid::TensorGrid2D, k)
    dx, dy = k
    Vx = derivative(funs.x, grid.xs, dx)
    Vy = derivative(funs.y, grid.ys, dy)
    _mat_mats_mat(Vx, funs.coefficients, Vy')
end

function derivative(funs::TensorFunctions2D, X::Array, Y::Array, k)
    dx, dy = k
    N = length(X)
    B = length(funs)
    Bx, By = length(funs.x), length(funs.y)
    Vx = derivative(funs.x, view(X, :), dx)
    Vy = derivative(funs.y, view(Y, :), dy)
    Vx = reshape(Vx, N, Bx, 1, 1)
    Vy = reshape(Vy, N, 1, By, 1)
    W = reshape(funs.coefficients, 1, Bx, By, B)
    reshape(sum(Vx .* Vy .* W; dims = (2, 3)), size(X)..., B)
end

(funs::TensorFunctions2D)(grid::TensorGrid2D) = derivative(funs, grid, (0, 0))
(funs::TensorFunctions2D)(X::AbstractArray, Y::AbstractArray) = derivative(funs, X, Y, (0, 0))

firstindex(funs::TensorFunctions2D) = 1
lastindex(funs::TensorFunctions2D) = size(funs.coefficients, 3)
length(funs::TensorFunctions2D) = size(funs.coefficients, 3)
function getindex(funs::TensorFunctions2D, idx::Integer)
    TensorFunction2D(funs.x, funs.y, funs.coefficients[:, :, idx])
end
function getindex(funs::TensorFunctions2D, ids::AbstractVector)
    TensorFunctions2D(funs.x, funs.y, funs.coefficients[:, :, ids])
end
function view(funs::TensorFunctions2D, idx::Integer)
    TensorFunction2D(funs.x, funs.y, view(funs.coefficients, :, :, idx))
end
function view(funs::TensorFunctions2D, ids::AbstractVector)
    TensorFunctions2D(funs.x, funs.y, view(funs.coefficients, :, :, ids))
end

function _mat_mats_mat(A, Bs, C)
    # Compute matrix * (batch of matrices) * matrix, i.e. Outᵢₗₙ = Aᵢⱼ Bsⱼₖₙ Cₖₗ
    I, J = size(A)
    K, L = size(C)
    N = size(Bs, 3)
    out = Array{Float64, 3}(undef, I, L, N)
    tmp = Array{Float64, 2}(undef, I, K)
    for n in 1:N
        # out[:, :, n] = A * Bs[:, :, n] * C
        mul!(tmp, A, view(Bs, :, :, n))
        mul!(view(out, :, :, n), tmp, C)
    end
    out
end

function flatten(funcs::TensorFunctions2D)
    fx = flatten(funcs.x)
    fy = flatten(funcs.y)
    coeffs = funcs.coefficients
    if fx isa Functions1D
        coeffs = _mat_mats_mat(fx.coefficients, coeffs, I(size(coeffs, 2)))
        fx = fx.basis
    end
    if fy isa Functions1D
        coeffs = _mat_mats_mat(I(size(coeffs, 1)), coeffs, fy.coefficients')
        fy = fy.basis
    end
    return TensorFunctions2D(fx, fy, coeffs)
end
