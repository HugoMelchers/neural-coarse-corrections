#=
An implementation of element-wise assembly for the diffusion and advection-diffusion equations.
Previously, I assembled the FEM systems globally by multiplying large sparse matrices containing
basis functions or their derivatives, and diagonal matrices containing PDE coefficients. However,
this is quite slow (much slower than the linear solve).
=#

"""
    sparsity_pattern(basis_1d)

Given a 1D B-spline basis, construct an all-zero sparse matrix whose sparsity pattern corresponds to
that of the Gram matrix of the 2D tensor product of this basis. For any PDE, the discretisation onto
this 2D tensor basis will have this sparsity pattern.

When assembling a PDE, it is useful to initialise the system matrix in this way instead of using
`spzeros`, as having the correct sparsity pattern from the start means that no re-allocations need
to be done on the system matrix, which greatly speeds up assembly.

`basis_1d` should be a `BSpline` basis with `sparse=true`.
"""
function sparsity_pattern(basis_1d)
    knots = unique(basis_1d.knots)
    midpoints = @views @. (knots[1:end-1] + knots[2:end]) / 2
    basis_values = basis_1d(midpoints)
    gram_1d = basis_values' * basis_values
    gram_2d = kron(gram_1d, gram_1d)
    gram_2d.nzval .= 0.0
    gram_2d
end

"""
    interior(basis_1d, quad)

Returns a pair (global_quadrature, elements), where `global_quadrature` consists of the nodes and
weights of a quadrature over the entire domain, and `elements` is an iterator over each element of
the discretisation. The elements are rectangles, namely the Cartesian product of knot spans in the
x- and y-axes. For each element, the iterator returns a NamedTuple containing information about the
element, including

- the width and height of the rectangular element
- a list of the global IDs of the basis functions supported on this element
- the values, derivatives, and Laplacians of those basis functions
- the nodes (a tensor product of nodes_x and nodes_y) and weights of the quadrature on the element
- a list of the indices of the nodes and weights of the `global_quadrature` rules corresponding to
  the current element. These are useful when a function has already been sampled on the global
  quadrature rule, so that to get the values on the current element one only needs to index into a
  larger array
"""
function interior(basis_1d, quad)
    nr_dofs_1d = length(basis_1d)

    # A B-spline basis with `n` segments and degree `p` will have `n + p` basis functions
    # Therefore the number of segments is length(basis) - p
    p = basis_1d.degree
    knots = basis_1d.knots
    nr_segments = nr_dofs_1d - p

    weights_2d = zeros(quad^2)

    vals_1d_x = zeros(quad, p + 1)
    vals_1d_y = zeros(quad, p + 1)
    vals_1d_x_deriv = zeros(quad, p + 1)
    vals_1d_y_deriv = zeros(quad, p + 1)
    vals_1d_x_deriv2 = zeros(quad, p + 1)
    vals_1d_y_deriv2 = zeros(quad, p + 1)

    basis = zeros(quad^2, (p + 1)^2)
    basis_dx = zeros(quad^2, (p + 1)^2)
    basis_dy = zeros(quad^2, (p + 1)^2)
    basis_dxx = zeros(quad^2, (p + 1)^2)
    basis_dyy = zeros(quad^2, (p + 1)^2)
    basis_lap = zeros(quad^2, (p + 1)^2)

    # Evaluate all basis functions and derivatives globally first, then index into them later
    knots_uniq = unique(knots)
    nodes_global, weights_global = repeated_gauss_legendre(knots_uniq, quad)
    u_1d_global = basis_1d(nodes_global)
    du_1d_global = derivative(basis_1d, nodes_global)
    d2u_1d_global = derivative(basis_1d, nodes_global, 2)

    ids = zeros(Int, p + 1, p + 1)

    global_quadrature = (;
        nodes = nodes_global,
        weights = weights_global,
    )

    iter = Iterators.map(Iterators.product(1:nr_segments, 1:nr_segments)) do (i, j)
        supp_x = i:i+p
        supp_y = j:j+p

        x0, x1 = knots[i+p], knots[i+p+1]
        y0, y1 = knots[j+p], knots[j+p+1]

        nodes_x = quad * (i - 1) .+ (1:quad)
        nodes_y = quad * (j - 1) .+ (1:quad)

        weights_x = view(weights_global, nodes_x)
        weights_y = view(weights_global, nodes_y)
        kron!(weights_2d, weights_y, weights_x)

        vals_1d_x .= view(u_1d_global, nodes_x, supp_x)
        vals_1d_y .= view(u_1d_global, nodes_y, supp_y)
        vals_1d_x_deriv .= view(du_1d_global, nodes_x, supp_x)
        vals_1d_y_deriv .= view(du_1d_global, nodes_y, supp_y)
        vals_1d_x_deriv2 .= view(d2u_1d_global, nodes_x, supp_x)
        vals_1d_y_deriv2 .= view(d2u_1d_global, nodes_y, supp_y)

        kron!(basis, vals_1d_y, vals_1d_x)
        kron!(basis_dx, vals_1d_y, vals_1d_x_deriv)
        kron!(basis_dy, vals_1d_y_deriv, vals_1d_x)
        kron!(basis_dxx, vals_1d_y, vals_1d_x_deriv2)
        kron!(basis_dyy, vals_1d_y_deriv2, vals_1d_x)
        basis_lap .= basis_dxx .+ basis_dyy

        ids .= supp_x .+ (supp_y' .- 1) .* nr_dofs_1d

        (;
            dof_ids = view(ids, :),
            nodes_x = view(nodes_global, nodes_x),
            nodes_y = view(nodes_global, nodes_y),
            x_ids = nodes_x,
            y_ids = nodes_y,
            weights = weights_2d,
            u = basis,
            u_dx = basis_dx,
            u_dy = basis_dy,
            u_lap = basis_lap,
            width = x1 - x0,
            height = y1 - y0,
        )
    end

    global_quadrature, iter
end

"""
    boundary_faces(basis_1d, quad, part)

TBW
"""
function boundary_faces(basis_1d, quad, part)
    @assert part in [:left, :right, :bottom, :top]
    is_x = part == :left || part == :right
    is_low = part == :left || part == :bottom
    nodes, weights = gausslegendre(quad)
    n_dof_1d = length(basis_1d)

    # A B-spline basis with `n` segments and degree `p` will have `n + p` basis functions
    # Therefore the number of segments is length(basis) - p
    p = basis_1d.degree
    knots = basis_1d.knots
    n_segments = n_dof_1d - p

    weights_2d = zeros(quad^2)

    vals_1d = zeros(quad, p + 1)
    vals_1d_deriv = zeros(quad, p + 1)

    basis = zeros(quad, (p + 1)^2)
    basis_dx = zeros(quad, (p + 1)^2)
    basis_dy = zeros(quad, (p + 1)^2)
    basis_dn = zeros(quad, (p + 1)^2)

    # Evaluate all basis functions and derivatives globally first, then index into them later
    knots_uniq = unique(knots)
    nodes_global, weights_global = repeated_gauss_legendre(knots_uniq, quad)
    u_1d_global = basis_1d(nodes_global)
    du_1d_global = derivative(basis_1d, nodes_global)
    nb = is_low ? [0.0] : [1.0]
    wb = [1.0]
    u_1d_b = basis_1d(nb)
    u_1d_db = derivative(basis_1d, nb)
    if is_low
        u_1d_b = u_1d_b[:, 1:p+1]
        u_1d_db = u_1d_db[:, 1:p+1]
    else
        u_1d_b = u_1d_b[:, end-p:end]
        u_1d_db = u_1d_db[:, end-p:end]
    end
    u_1d_dn = if is_low
        .-u_1d_db
    else
        u_1d_db
    end
    normal = is_low ? -1.0 : 1.0

    ids = zeros(Int, p + 1, p + 1)

    global_data = (; nodes = nodes_global, weights = weights_global)
    iter = Iterators.map(1:n_segments) do k
        i = is_x ? (is_low ? 1 : n_segments) : k
        j = !is_x ? (is_low ? 1 : n_segments) : k
        supp = k:k+p
        supp_x = i:i+p
        supp_y = j:j+p

        nodes = quad * (k - 1) .+ (1:quad)
        weights = view(weights_global, nodes)
        weights_x = is_x ? wb : weights
        weights_y = !is_x ? wb : weights

        vals_1d .= view(u_1d_global, nodes, supp)
        vals_1d_deriv .= view(du_1d_global, nodes, supp)

        vals_1d_x = is_x ? u_1d_b : vals_1d
        vals_1d_y = !is_x ? u_1d_b : vals_1d
        vals_1d_x_deriv = is_x ? u_1d_db : vals_1d_deriv
        vals_1d_y_deriv = !is_x ? u_1d_db : vals_1d_deriv

        kron!(basis, vals_1d_y, vals_1d_x)
        kron!(basis_dx, vals_1d_y, vals_1d_x_deriv)
        kron!(basis_dy, vals_1d_y_deriv, vals_1d_x)
        if is_x
            basis_dn .= basis_dx
        else
            basis_dn .= basis_dy
        end
        if is_low
            basis_dn .= .-basis_dn
        end

        ids .= supp_x .+ (supp_y' .- 1) .* n_dof_1d

        (;
            part = part,
            normal,
            dof_ids = view(ids, :),
            nodes = view(nodes_global, nodes),
            p_ids = nodes,
            weights,
            u = basis,
            u_dn = basis_dn,
        )
    end

    global_data, iter
end
