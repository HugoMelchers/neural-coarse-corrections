#=
Element-wise assembly for PDEs discretised on a 3D tensor-product B-spline basis.

This is the 3D counterpart of `../assembly.jl`. The main difference is that element matrices are not
computed by multiplying tables of the 3D basis functions on the quadrature nodes (as in 2D), since
in 3D these tables become large: (quad³) × (p+1)³. Instead, all integrals are computed with sum
factorisation, i.e. by contracting over the quadrature nodes in x, y, and z one at a time, using
the 1D basis function values only. For example, for an integrand of the form

    ∂ₓφᵢ(x, y, z) θ(x, y, z) ∂ₓφⱼ(x, y, z)

where φᵢ(x, y, z) = φₐ(x)φ_b(y)φ_c(z) and φⱼ(x, y, z) = φₐ₂(x)φ_b₂(y)φ_c₂(z), the element matrix is

    K[a, b, c, a2, b2, c2] = Σ_{qx, qy, qz} Dx[qx, a] Dx[qx, a2] Vy[qy, b] Vy[qy, b2]
                                             Vz[qz, c] Vz[qz, c2] θW[qx, qy, qz]

which is evaluated as three successive contractions over qx, qy, and qz.

Faces of the domain are treated in the same way, by using a "quadrature" with a single node (with
weight 1) in the direction normal to the face.

Throughout, basis functions are ordered as x-varies-first, then y, then z, i.e. the 3D basis
function with 1D indices (i, j, k) has index i + (j - 1) n + (k - 1) n².

Note that the tensor contractions are done with Tullio.jl and with `threads=false`, i.e. single-
threaded. I don't use multi-threading here as the dataset generation is perfectly parallelisable on
the level of individual PDEs, so each individiual assemble and solve can be done on just one thread.
=#

import SparseArrays: SparseMatrixCSC, rowvals, nonzeros, nzrange

"""
    sparsity_pattern_3d(basis_1d)

3D equivalent of `sparsity_pattern`: an all-zero sparse matrix with the sparsity pattern of the
Gram matrix of the 3D tensor product of `basis_1d`, which should be a `BSpline` with `sparse=true`.
"""
function sparsity_pattern_3d(basis_1d)
    knots = unique(basis_1d.knots)
    midpoints = @views @. (knots[1:(end-1)] + knots[2:end]) / 2
    basis_values = basis_1d(midpoints)
    gram_1d = basis_values' * basis_values
    gram_3d = kron(gram_1d, gram_1d, gram_1d)
    gram_3d.nzval .= 0.0
    gram_3d
end

"""
    add_to_sparse!(K, ids, K_el)

Perform `K[ids, ids] .+= K_el` for a sparse matrix `K` without changing its sparsity pattern. All
entries `K[ids, ids]` must already be part of the sparsity pattern (which should be the case as long
as `K` is produced using `sparsity_pattern_3d`), and `ids` must be sorted in increasing order (which
is the case for the `dof_ids` of the 3D elements).
"""
function add_to_sparse!(K::SparseMatrixCSC, ids, K_el)
    rows = rowvals(K)
    vals = nonzeros(K)
    for (b, col) in enumerate(ids)
        r = nzrange(K, col)
        pos = first(r)
        for (a, row) in enumerate(ids)
            # Both `ids` and the row indices of each column are sorted, so we can walk along them
            while rows[pos] != row
                pos += 1
                pos <= last(r) || error("add_to_sparse!: ($row, $col) not in sparsity pattern")
            end
            vals[pos] += K_el[a, b]
        end
    end
    K
end

"""
    element_buffers(p, nq)

Allocate the temporary arrays needed by `add_element_matrix!` and `add_element_vector!`, for
B-splines of degree `p` and an element with `nq = (nqx, nqy, nqz)` quadrature nodes per direction.
"""
function element_buffers(p, nq)
    m = p + 1
    _nqx, nqy, nqz = nq
    (;
        T1 = zeros(m, m, nqy, nqz),
        T2 = zeros(m, m, m, m, nqz),
        t1 = zeros(m, nqy, nqz),
        t2 = zeros(m, m, nqz),
    )
end

"""
    add_element_matrix!(K_el, buffers, α, A, A2, B, B2, C, C2, W)

Compute, with sum factorisation,

    K_el[a, b, c, a2, b2, c2] += α Σ_{qx,qy,qz} A[qx, a] A2[qx, a2] B[qy, b] B2[qy, b2]
                                                C[qz, c] C2[qz, c2] W[qx, qy, qz]

Here, the matrices `A`, `B`, `C` contain the values (or derivatives) of the 1D test functions on
the quadrature nodes in x, y, z, and `A2`, `B2`, `C2` the same for the trial functions. `W`
contains the quadrature weights, multiplied by any PDE coefficients.
"""
function add_element_matrix!(K_el, buffers, α, A, A2, B, B2, C, C2, W)
    (; T1, T2) = buffers
    @tullio T1[a, a2, qy, qz] = A[qx, a] * A2[qx, a2] * W[qx, qy, qz] threads = false
    @tullio T2[a, a2, b, b2, qz] = B[qy, b] * B2[qy, b2] * T1[a, a2, qy, qz] threads = false
    @tullio(
        K_el[a, b, c, a2, b2, c2] += α * C[qz, c] * C2[qz, c2] * T2[a, a2, b, b2, qz],
        threads = false,
    )
    K_el
end

"""
    add_element_vector!(b_el, buffers, α, A, B, C, W)

Compute, with sum factorisation,

    b_el[a, b, c] += α Σ_{qx,qy,qz} A[qx, a] B[qy, b] C[qz, c] W[qx, qy, qz]
"""
function add_element_vector!(b_el, buffers, α, A, B, C, W)
    (; t1, t2) = buffers
    @tullio t1[a, qy, qz] = A[qx, a] * W[qx, qy, qz] threads = false
    @tullio t2[a, b, qz] = B[qy, b] * t1[a, qy, qz] threads = false
    @tullio b_el[a, b, c] += α * C[qz, c] * t2[a, b, qz] threads = false
    b_el
end

_tensor_weights!(W, wx, wy, wz) =
    @tullio W[i, j, k] = wx[i] * wy[j] * wz[k] threads = false

function _dof_ids!(ids, supp_x, supp_y, supp_z, n)
    ids .= supp_x .+ (supp_y' .- 1) .* n .+ (reshape(supp_z, 1, 1, :) .- 1) .* n^2
end

"""
    interior_3d(basis_1d, quad)

3D equivalent of `interior`. Returns a pair (global_quadrature, elements), where
`global_quadrature` contains the 1D nodes and weights of a composite Gauss-Legendre rule (with
`quad` nodes per knot span) on the whole interval, and `elements` iterates over all elements (cubes
formed by a knot span in each of x, y, and z). For each element, the iterator returns a NamedTuple
containing

- `dof_ids`: the global IDs of the (p+1)³ basis functions supported on this element
- `node_ids`: a tuple (x_ids, y_ids, z_ids) of indices into the global 1D quadrature nodes, so
  that for a function `F` sampled on `nodes ⊗ nodes ⊗ nodes`, `view(F, node_ids...)` gives its
  values on the element's quadrature nodes
- `weights`: a quad × quad × quad array of quadrature weights
- `u`, `du`: tuples of 1D basis function values and derivatives (each a quad × (p+1) matrix) in
  the x, y, and z directions

Note that the arrays in the returned NamedTuple are re-used between iterations.
"""
function interior_3d(basis_1d, quad)
    nr_dofs_1d = length(basis_1d)
    p = basis_1d.degree
    knots = basis_1d.knots
    nr_segments = nr_dofs_1d - p

    # Evaluate all basis functions and derivatives globally first, then index into them later
    knots_uniq = unique(knots)
    nodes_global, weights_global = repeated_gauss_legendre(knots_uniq, quad)
    u_1d_global = _dense(basis_1d(nodes_global))
    du_1d_global = _dense(derivative(basis_1d, nodes_global))

    global_quadrature = (; nodes = nodes_global, weights = weights_global)

    vals = ntuple(_ -> zeros(quad, p + 1), 3)
    derivs = ntuple(_ -> zeros(quad, p + 1), 3)
    weights_3d = zeros(quad, quad, quad)
    ids = zeros(Int, p + 1, p + 1, p + 1)

    product = Iterators.product(1:nr_segments, 1:nr_segments, 1:nr_segments)
    iter = Iterators.map(product) do segments
        supp = map(s -> s:(s+p), segments)
        node_ids = map(s -> quad * (s - 1) .+ (1:quad), segments)
        for d in 1:3
            vals[d] .= view(u_1d_global, node_ids[d], supp[d])
            derivs[d] .= view(du_1d_global, node_ids[d], supp[d])
        end
        _tensor_weights!(weights_3d, map(n -> view(weights_global, n), node_ids)...)
        _dof_ids!(ids, supp..., nr_dofs_1d)

        (;
            dof_ids = view(ids, :),
            node_ids,
            weights = weights_3d,
            u = vals,
            du = derivs,
        )
    end

    global_quadrature, iter
end

const FACES_3D = (:left, :right, :front, :back, :bottom, :top)

"""
    face_axis(part)

For a face of the unit cube, return the pair (axis, is_low), where `axis` is the direction normal
to the face (1, 2, 3 for x, y, z) and `is_low` is whether the face lies at 0 (rather than 1). The
faces are `:left`/`:right` (x = 0 / x = 1), `:front`/`:back` (y = 0 / y = 1), and `:bottom`/`:top`
(z = 0 / z = 1).
"""
function face_axis(part)
    idx = findfirst(==(part), FACES_3D)
    @assert idx !== nothing "unknown face $part, expected one of $FACES_3D"
    (idx + 1) ÷ 2, isodd(idx)
end

"""
    boundary_faces_3d(basis_1d, quad, part)

3D equivalent of `boundary_faces`. Returns a pair (global_data, faces), where `faces` iterates over
the elements on the boundary face `part` (see `face_axis`), and `global_data` contains

- `nodes`, `weights`: the global 1D quadrature rule used in the two tangential directions
- `grid`: a `TensorGrid3D` of all quadrature nodes on the face, with a single node in the normal
  direction. Evaluating a 3D function on this grid gives an array with a singleton dimension.
- `face_grid`: the same grid as a `TensorGrid2D` of only the tangential directions, for
  evaluating 2D functions defined on the face. Reshaping the result to `size_3d` makes it
  compatible with `node_ids`.
- `size_3d`: the size of arrays sampled on `grid`.

Each face element is a NamedTuple with the same fields as in `interior_3d`, except that instead of
`du` it contains `u_dn`, the outward normal derivative of the 1D basis functions in direction `axis`.
The arrays in direction `axis` have a single quadrature node (with weight 1).
"""
function boundary_faces_3d(basis_1d, quad, part)
    axis, is_low = face_axis(part)
    nr_dofs_1d = length(basis_1d)
    p = basis_1d.degree
    knots = basis_1d.knots
    nr_segments = nr_dofs_1d - p

    knots_uniq = unique(knots)
    nodes_global, weights_global = repeated_gauss_legendre(knots_uniq, quad)
    u_1d_global = _dense(basis_1d(nodes_global))

    # Values and outward normal derivatives on the boundary point in the normal direction. Only the
    # first or last p + 1 basis functions are supported in the first or last knot span
    x_b = is_low ? knots[1] : knots[end]
    normal = is_low ? -1.0 : 1.0
    supp_b = is_low ? (1:(p+1)) : ((nr_dofs_1d-p):nr_dofs_1d)
    u_b = _dense(basis_1d([x_b]))[:, supp_b]
    u_dn = normal .* _dense(derivative(basis_1d, [x_b]))[:, supp_b]
    segment_b = is_low ? 1 : nr_segments

    tangential = filter(!=(axis), (1, 2, 3))
    size_3d = ntuple(d -> d == axis ? 1 : length(nodes_global), 3)
    coords = ntuple(d -> d == axis ? [x_b] : nodes_global, 3)
    grid = coords[1] ⊗ coords[2] ⊗ coords[3]
    face_grid = coords[tangential[1]] ⊗ coords[tangential[2]]
    global_data = (; nodes = nodes_global, weights = weights_global, grid, face_grid, size_3d)

    vals = ntuple(d -> d == axis ? u_b : zeros(quad, p + 1), 3)
    weights_3d = zeros(ntuple(d -> d == axis ? 1 : quad, 3))
    ids = zeros(Int, p + 1, p + 1, p + 1)
    one_weight = [1.0]

    iter = Iterators.map(Iterators.product(1:nr_segments, 1:nr_segments)) do (k1, k2)
        segments = ntuple(3) do d
            d == axis ? segment_b : (d == tangential[1] ? k1 : k2)
        end
        supp = map(s -> s:(s+p), segments)
        node_ids = ntuple(d -> d == axis ? (1:1) : quad * (segments[d] - 1) .+ (1:quad), 3)
        for d in tangential
            vals[d] .= view(u_1d_global, node_ids[d], supp[d])
        end
        weights_1d = ntuple(d -> d == axis ? one_weight : view(weights_global, node_ids[d]), 3)
        _tensor_weights!(weights_3d, weights_1d...)
        _dof_ids!(ids, supp..., nr_dofs_1d)

        (;
            part,
            axis,
            normal,
            dof_ids = view(ids, :),
            node_ids,
            weights = weights_3d,
            u = vals,
            u_dn,
        )
    end

    global_data, iter
end
