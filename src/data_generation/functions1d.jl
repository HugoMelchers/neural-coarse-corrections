import Base: length, getindex, firstindex, lastindex, view, getindex

# Single function in 1D
struct Function1D{T, TF <: AbstractVector}
    basis::T
    coefficients::TF

    function Function1D(basis::T, coefficients::TF) where {T, TF}
        @assert length(basis) == length(coefficients)
        new{T, TF}(basis, coefficients)
    end
end

function derivative(fun::Function1D, xs::AbstractArray, k = 1)
    xs_flat = view(xs, :)
    basis = derivative(fun.basis, xs_flat, k)
    out = basis * fun.coefficients
    reshape(out, size(xs))
end

(fun::Function1D)(xs::Array) = derivative(fun, xs, 0)

# Multiple functions in 1D
struct Functions1D{T, TF <: AbstractMatrix}
    basis::T
    coefficients::TF

    function Functions1D(basis::T, coefficients::TF) where {T, TF}
        @assert length(basis) == size(coefficients, 1)
        new{T, TF}(basis, coefficients)
    end
end

firstindex(funs::Functions1D) = 1
lastindex(funs::Functions1D) = size(funs.coefficients, 2)
length(funs::Functions1D) = size(funs.coefficients, 2)
getindex(funs::Functions1D, idx::Integer) = Function1D(funs.basis, funs.coefficients[:, idx])

function getindex(funs::Functions1D, ids::AbstractVector)
    Functions1D(funs.basis, funs.coefficients[:, ids])
end

view(funs::Functions1D, idx::Integer) = Function1D(funs.basis, view(funs.coefficients, :, idx))

function view(funs::Functions1D, ids::AbstractVector)
    Functions1D(funs.basis, view(funs.coefficients, :, ids))
end

function derivative(funs::Functions1D, xs::AbstractArray, k = 1)
    xs_flat = view(xs, :)
    basis = derivative(funs.basis, xs_flat, k)
    out = basis * funs.coefficients
    reshape(out, size(xs)..., length(funs))
end

(funs::Functions1D)(xs::Array) = derivative(funs, xs, 0)

function flatten(funcs::Functions1D)
    # If we have a linear combination of linear combinations of functions, just express the functions
    # in terms of the original basis directly.
    if funcs.basis isa Functions1D
        inner = flatten(funcs.basis)
        Functions1D(inner.basis, inner.coefficients * funcs.coefficients)
    else
        funcs
    end
end
