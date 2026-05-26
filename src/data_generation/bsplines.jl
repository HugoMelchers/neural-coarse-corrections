#=
Implementation of 1D B-spline bases.

Features:
- Arbitrary degree splines
- Arbitrary knot vectors
- Support for derivatives
- Support for outputting sparse arrays
- Accessible as a `BSpline` struct as well as a `bspline` function
- For discontinuous bases: choice between left and right continuity

Limitations:
- Functions near the boundary are NOT polynomial-extended
- No support for GPUs
- Currently no support for evaluating a B-spline basis at a single point. This has to be done as
  basis(x::Number) = basis([x])[1, :]
- Currently no support for single B-spline functions (i.e. the output is always a matrix)
- A slight inefficiency for closed knot vectors?

Written by Hugo Melchers, version 2025-03-17.
=#

import Base: length, getindex, firstindex, lastindex, view
using SparseArrays
import FastGaussQuadrature: gausslegendre

struct BSpline{T, TV <: AbstractVector{T}}
    # The degree of the BSpline basis
    degree::Int

    # The knot vector. Note that here, the knot vector is *not* treated as open by default, meaning
    # that for open knot vectors the first and last knots have to be repeated in this vector.
    knots::TV

    # Whether to output a SparseMatrixCSC instead of a regular dense Matrix
    sparse::Bool
end

flatten(funcs::BSpline) = funcs

"""
    BSpline(lims::NTuple{2}, n_knot_spans::Int, degree::Int; cont=nothing)

Construct a BSpline basis supported on [lims[1], lims[2]] with a uniform open knot vector.

- `n_knot_spans` is the number of knot spans of the basis. For example, if lims = (0, 1) and
  n_knot_spans = 5, then the knot vector will be [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
- `degree` is the degree of the B-spline basis
- `cont` is the continuity of the B-spline basis, i.e. how many derivatives of the basis exist on
  the knots. By default, cont = degree - 1 (the highest legal value), meaning maximal smoothness of
  the B-spline basis. Lowering `cont` increases the number of basis functions. For example, setting
  `cont = 0` produces continuous, but not continuously differentiable, basis functions, similar
  to the basis functions in classical FEM. Setting `cont = -1` (the lowest legal value) produces
  discontinuous basis functions.

The resulting `basis` can be called on an array of numbers `xs`, yielding a matrix A = basis(xs)
where A[i,j] is the j-th basis function evaluated at xs[i]. Alternatively calling A = basis(xs;
sparse=true) yields a sparse matrix for A. Calling `derivative(basis, xs, k)` or `derivative(basis,
xs, k; sparse=true)` produces the `k`-th derivative of the basis functions.

Notes:
- here, the knot vector is open, meaning that the first and last knots are repeated `degree + 1`
  times. This produces a basis that is nodal on the two endpoints
- All B-spline functions are equal to zero outside of the domain `lims`
- For discontinuous basis functions (so cont=-1 or degree=0), the functions are right-continuous on
  the internal knots. In the `bspline` function, this can be changed to left-continuity, but not in
  the BSpline struct.
"""
function BSpline(
    lims::NTuple{2},
    n_knot_spans::Int,
    degree::Int;
    cont = nothing,
    sparse = false,
)
    # Multiplicity of each internal knot
    mult = if cont !== nothing
        @assert -1 ≤ cont && cont ≤ degree - 1
        degree - cont
    else
        1
    end
    a, b = lims
    knots = range(a, b, n_knot_spans + 1)[2:(end-1)]
    knots = repeat(knots; inner = mult)
    knots_open = vcat(fill(a, degree + 1), knots, fill(b, degree + 1))
    BSpline(degree, knots_open, sparse)
end

firstindex(fs::BSpline) = 1
lastindex(fs::BSpline) = length(fs)

length(fs::BSpline) = length(fs.knots) - 1 - fs.degree

function _get_knot_range(fs::BSpline, func_range)
    # Slicing is only supported for integer ranges with step 1 (i.e. continuous ranges)
    @assert func_range isa AbstractRange{Int}
    @assert step(func_range) == 1
    @assert func_range[1] >= 1
    @assert func_range[end] <= length(fs)
    skip_left = func_range[1] - 1
    skip_right = length(fs) - func_range[end]
    n = length(fs.knots)
    return (1+skip_left):(n-skip_right)
end

function getindex(fs::BSpline, range)
    r = _get_knot_range(fs, range)
    return BSpline(fs.degree, fs.knots[r], fs.sparse)
end

function view(fs::BSpline, range)
    r = _get_knot_range(fs, range)
    return BSpline(fs.degree, view(fs.knots, r), fs.sparse)
end

(fs::BSpline)(xs) = bspline(fs.knots, fs.degree, xs; sparse = fs.sparse, open = false)
derivative(fs::BSpline, xs, k = 1) =
    bspline(fs.knots, fs.degree, xs; sparse = fs.sparse, open = false, deriv = k)

"""
    eval_bspline_parts(knots, degree, xs; deriv=0)

Evaluate a bspline basis, producing a compact representation of the output values.

- `knots`: the knot vector
- `degree`: the degree of the splines
- `xs`: the points to evaluate the functions at
- `deriv=0`: the derivative to take

The output is represented as a list of integers `i`, and a matrix of Float64's `v`. For each `x`,
these values are:
- `i` is the index of the knot span containing `x`
- `v` is an array of `deg + 1` values, containing the values of the basis functions supported there

Knot vectors are implicitly treated as open, meaning that the first and last knots are repeated
`degree + 1` times. This is done because in that case, all points `x` (except the knots themselves)
lie in the support of exactly `degree + 1` basis functions, meaning that `v` is a perfectly
rectangular array. If the knot vector isn't open, then the first and last `degree` basis functions
should be removed from the output.
"""
function eval_bspline_parts(knots, degree, xs; deriv = 0, right_cont = true)
    # For each x, k is the corresponding segment that x lies in
    @assert deriv >= 0
    span_indices = if right_cont
        searchsortedlast.(Ref(knots), xs)
    else
        searchsortedfirst.(Ref(knots), xs) .- 1
    end
    nr_x = length(xs)
    if deriv > degree
        return span_indices, zeros(nr_x, degree + 1)
    end

    T = eltype(xs)

    # clamp!(i_x, 1, length(knots) - 1)
    first_nonempty_span = findlast(==(knots[1]), knots)
    last_nonempty_span = findfirst(==(knots[end]), knots) - 1
    clamp!(span_indices, first_nonempty_span, last_nonempty_span)

    # In total, a point can have deg + 1 basis functions that are non-zero there
    values = zeros(T, nr_x, degree + 1)
    vals_tmp = zeros(T, nr_x, degree + 1)
    values[:, 1] .= ifelse.(xs .>= knots[1] .&& xs .<= knots[end], one(T), zero(T))
    nr_knots = length(knots)
    α = zeros(T, nr_x, degree)

    J = span_indices .+ range(0, degree - 1)'
    for p in 1:degree
        js = view(J, :, 1:p)
        ks = getindex.(Ref(knots), clamp.(range(2 - p, nr_knots + p - 1), 1, nr_knots))
        l = nr_knots + 2p - 2
        diffs = max.(view(ks, (p+1):l) - view(ks, 1:(l-p)), eps())
        α_view = view(α, :, 1:p)
        v_view = view(values, :, 1:p)
        if (degree - p) ≥ deriv
            α_view .= (xs .- getindex.(Ref(ks), js)) ./ getindex.(Ref(diffs), js)
            view(vals_tmp, :, 1:p) .= (1 .- α_view) .* v_view
            view(vals_tmp, :, 2:(p+1)) .+= α_view .* v_view
        else
            # Differentiation step
            α_view .= p ./ getindex.(Ref(diffs), js)
            view(vals_tmp, :, 1:p) .= -α_view .* v_view
            view(vals_tmp, :, 2:(p+1)) .+= α_view .* v_view
        end
        values, vals_tmp = vals_tmp, values
    end
    span_indices, values
end

"""
    bspline(knots, degree, xs; sparse=false, open=false, deriv=0)

Evaluate a bspline basis.

- `knots`: the knot vector
- `degree`: the spline degree
- `xs`: the vector of points to evaluate the functions at
- `sparse`: whether or not to return the output as sparse array (default: false)
- `open`: if set to true, treat the knot vector as open, by repeating the first and last knots
  (default: false)
- `deriv`: instead of the values, compute the `deriv`-th derivative (default: 0)
"""
function bspline(
    knots,
    degree,
    xs;
    sparse = false,
    open = false,
    deriv = 0,
    right_cont = true,
)
    T = eltype(xs)
    nr_x = length(xs)
    span_indices, values = eval_bspline_parts(knots, degree, xs; deriv, right_cont)
    nr_knot_spans = length(knots) - 1
    if open
        # Pretend that the first and last knots are repeated an additional 'degree' times
        span_indices .+= degree
        nr_knot_spans += 2 * degree
    end
    nf = nr_knot_spans - degree
    out = if sparse
        spzeros(T, nr_x, nf)
    else
        zeros(T, nr_x, nf)
    end
    sel_i = BitVector(undef, nr_x)
    # For each knot span, find the basis functions that are supported there, their indices in the
    # output array, and the indices in the array `v`, then copy those values over. For knot spans
    # away from the edges, if x lies in knot span i then the basis functions supported there have
    # indices i - degree, i - degree + 1, ..., i. However, near the edges the first few, or last few
    # (or both) of this range may not exist. So the min() and max() are to only select the range
    for knot_span in 1:nr_knot_spans
        idx_1 = max(1, degree + 2 - knot_span)
        idx_2 = min(degree + 1, nf + 1 - knot_span + degree)
        @. sel_i = span_indices == knot_span
        sel_j = idx_1:idx_2
        out[sel_i, sel_j.+(knot_span-degree-1)] .= view(values, sel_i, sel_j)
    end

    out
end

function repeated_gauss_legendre(knots, n_per_span)
    n0, w0 = gausslegendre(n_per_span)
    mids = @views @. (knots[1:(end-1)] + knots[2:end]) / 2
    halfwidths = @views @. (knots[2:end] - knots[1:(end-1)]) / 2
    n = mids' .+ n0 .* halfwidths'
    w = w0 .* halfwidths'
    return n[:], w[:]
end
