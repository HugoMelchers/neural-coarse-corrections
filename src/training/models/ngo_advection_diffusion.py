import torch

from src.training.subnets import TensorBasis, BSpline


def data_ngo_assemble_advdiff(batch, basis, config):
    assert isinstance(basis, TensorBasis)
    assert isinstance(basis.f_x, BSpline)
    assert isinstance(basis.f_y, BSpline)
    sga = config.get("sga", None)
    scale = config.get("scale", False)
    gamma = config.get("gamma", None)
    inflow_stab = config.get("inflow_stab", False)
    supg = config.get("supg", False)
    assert sga in [None, "bspline"]

    nodes_x, weights_x = batch["nodes_x"], batch["weights_x"]
    nodes_y, weights_y = batch["nodes_y"], batch["weights_y"]
    dtype, device = nodes_x.dtype, nodes_x.device
    f_x = basis.f_x
    f_y = basis.f_y if basis.f_y is not None else basis.f_x
    basis_x = f_x(nodes_x)
    basis_y = f_y(nodes_y)
    nodes_b = torch.tensor([0.0, 1.0], dtype=dtype, device=device)
    weights_b = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
    normals = torch.tensor([-1.0, 1.0], dtype=dtype, device=device)
    basis_x_b = f_x(nodes_b)
    basis_y_b = f_y(nodes_b)
    d_basis_x = f_x.derivative(nodes_x)
    d_basis_y = f_y.derivative(nodes_y)
    d_basis_x_b = f_x.derivative(nodes_b)
    d_basis_y_b = f_y.derivative(nodes_b)

    theta = batch["theta"]
    f = batch["f"]
    g = batch["dirichlet:g"]
    eta = batch["neumann:eta"]
    v_x = batch["v_x"]
    v_y = batch["v_y"]

    scale_factor = None
    if scale not in [False, None]:
        # Scale theta such that its average equals 1
        # Then integrate theta - 1 against the basis, and also scale the right-hand-side vector
        theta_avg = torch.einsum("Bxy,x,y->B", theta, weights_x, weights_y)
        if scale == True or scale == "theta":
            # Scale so that theta averages 1
            scale_factor = 1 / theta_avg
            theta = theta * scale_factor[:, None, None] - 1
            v_x = v_x * scale_factor[:, None, None]
            v_y = v_y * scale_factor[:, None, None]
        elif scale == "theta+v":
            # Scale so that theta + |v_x| + |v_y| / 3 averages 1
            vx_avg = torch.einsum("Bxy,x,y->B", v_x.abs(), weights_x, weights_y)
            vy_avg = torch.einsum("Bxy,x,y->B", v_y.abs(), weights_x, weights_y)
            scale_factor = 3 / (theta_avg + vx_avg + vy_avg)
            theta = theta * scale_factor[:, None, None]
            v_x = v_x * scale_factor[:, None, None]
            v_y = v_y * scale_factor[:, None, None]
        else:
            raise ValueError(f"Illegal scaling type '{scale}'")

    # Matrix input: [⟨w, θ⟩_Ω, ⟨w, v_x⟩_Ω, ⟨w, v_y⟩_Ω]
    K_theta = torch.einsum("ix,x,Bxy,y,jy->Bij", basis_x, weights_x, theta, weights_y, basis_y)
    K_vx = torch.einsum("ix,x,Bxy,y,jy->Bij", basis_x, weights_x, v_x, weights_y, basis_y)
    K_vy = torch.einsum("ix,x,Bxy,y,jy->Bij", basis_x, weights_x, v_y, weights_y, basis_y)
    # Stack the three B × nx × ny inputs into a single B × 3 × nx × ny input
    K = torch.stack((K_theta, K_vx, K_vy), dim=1)

    # Vector input: ⟨w, f⟩_Ω - ⟨∇w⋅n⃗, g⟩_ΓD + (C/h) ⟨w, g⟩_ΓD + ⟨w, η⟩_ΓN
    # Here C/h is the Nitsche stabilisation coefficient, which may be zero
    b = torch.einsum("ix,x,Bxy,y,jy->Bij", basis_x, weights_x, f, weights_y, basis_y)
    b = b - torch.einsum(
        "ix,x,x,Bxy,y,jy->Bij", d_basis_x_b, weights_b, normals, g, weights_y, basis_y
    )
    if gamma is not None:
        b = b + gamma * torch.einsum(
            "ix,x,Bxy,y,jy->Bij", basis_x_b, weights_b, g, weights_y, basis_y
        )

    if inflow_stab:
        # Inflows stabilisation term ⟨-(v/θ ⋅ n)₋ w, g⟩_Ω
        # Dirichlet boundaries are left/right, so we only need v_x
        v_x_d = batch["dirichlet:v_x"]
        vn_min = (v_x_d * normals[:, None]).clamp_max(0)
        theta_d = batch["dirichlet:theta"]
        b = b - torch.einsum(
            "ix,x,Bxy,y,jy->Bij", basis_x_b, weights_b, vn_min / theta_d * g, weights_y, basis_y
        )

    if supg:
        # SUPG term ∑ₑ ∫Ωₑ (τc⃗⋅∇w, f) dx
        taus = _get_supg_tau(basis, (nodes_x, weights_x), (nodes_y, weights_y), batch)
        factor1 = torch.einsum("Bxy,Bxy,kx,ly->Bxykl", taus, v_x, d_basis_x, basis_y)
        factor1 += torch.einsum("Bxy,Bxy,kx,ly->Bxykl", taus, v_y, basis_x, d_basis_y)
        b = b + torch.einsum("Bxykl,Bxy,x,y->Bkl", factor1, batch["f"], weights_x, weights_y)

    b = b + torch.einsum("ix,x,Bxy,y,jy->Bij", basis_x, weights_x, eta, weights_b, basis_y_b)
    return K, b[:, None], scale_factor


def _get_supg_tau(basis, quad_x, quad_y, batch):
    """
        _get_supg_tau(basis, quad_x, quad_y, batch)

    For a batch of advection-diffusion problems, compute the stabilisation parameter tau. Inputs:

    - `basis`: the TensorBasis of BSpline bases used as the trial basis
    - `quad_x, quad_y`: two tuples (nodes, weights) containing the quadrature rules in the x- and
      y-axes. The full quadrature rule over the domain is the tensor product of these two rules
    - batch: the batch of discretised problems

    Output: an array of size B × nx × ny with B the size of the batch, nx the number of quadrature
    points in the x-direction, and ny the number of quadrature points in the y-direction. This
    contains, for each problem in the batch, the value of the stabilisation parameter tau on each of
    the quadrature points. It is piecewise constant within each element.
    """
    n1, w1 = quad_x
    n2, w2 = quad_y
    lw_x, lw_y = _get_log_widths(basis)
    n_el_x = lw_x.numel()
    n_el_y = lw_y.numel()
    basis_tau_x = BSpline((0, 1), n_el_x, 0, log_widths=lw_x)
    basis_tau_y = BSpline((0, 1), n_el_y, 0, log_widths=lw_y)
    basis_tau_1 = basis_tau_x(n1)
    basis_tau_2 = basis_tau_y(n2)

    # Compute cell sizes as sqrt(dx * dy)
    dx = lw_x.exp()
    dy = lw_y.exp()
    dx = dx / dx.sum()
    dy = dy / dy.sum()
    h = (dx[:, None] * dy[None, :]).sqrt()[None, :, :]

    # The velocity field is constant (for now), so we can just sample the first point
    v_norm = batch["v_x"][:, 0, 0].hypot(batch["v_y"][:, 0, 0])[:, None, None]

    # For theta: compute averages over each cell
    theta = batch["theta"]
    th_avg = torch.einsum("Bxy,kx,ly,x,y->Bkl", theta, basis_tau_1, basis_tau_2, w1, w2) / h**2

    # Compute tau
    Pe = v_norm * h / th_avg
    p = basis.f_x.degree
    tau = h / (2 * v_norm).clamp_min(1e-6) * (Pe / (3 * p**2)).clamp_max(1)
    # Make tau a piecewise constant function using a degree-0 B-spline basis
    taus = torch.einsum("Bkl,kx,ly->Bxy", tau, basis_tau_1, basis_tau_2)
    return taus


def _get_log_widths(basis):
    assert isinstance(basis, TensorBasis)
    p_x = _get_log_widths_1d(basis.f_x)
    p_y = _get_log_widths_1d(basis.f_y)
    return p_x, p_y


def _get_log_widths_1d(basis):
    if isinstance(basis, BSpline):
        return basis.log_widths
    else:
        raise TypeError(f"_get_partition_1d got type {type(basis)}")


def _integrate(
    basis_data,
    field,
    *,
    part="interior",
    deriv_x_l=False,
    deriv_y_l=False,
    deriv_x_r=False,
    deriv_y_r=False,
):
    # Compute an integral of the form ∫ φᵢⱼ(x, y) f(x, y) φₖₗ(x, y) d(x, y), either over the
    # interior of the domain or a boundary. Since the basis is tensorised, we can write this
    # integral as ∫ φᵢⱼ(x, y) f(x, y) φₖₗ(x, y) d(x, y), which in a quadrature rule becomes
    # ∑ₘₙ φᵢ(xₘ) φⱼ(yₙ) f(xₘ, yₙ) φₖ(xₘ) φₗ(yₙ) wₘ wₙ, which I compute using torch.einsum.
    basis_data = basis_data[part]
    bx1 = basis_data["dx"] if deriv_x_l else basis_data["x"]
    bx2 = basis_data["dx"] if deriv_x_r else basis_data["x"]
    by1 = basis_data["dy"] if deriv_y_l else basis_data["y"]
    by2 = basis_data["dy"] if deriv_y_r else basis_data["y"]
    wx = basis_data["wx"]
    wy = basis_data["wy"]

    return torch.einsum("im,jn,Bmn,km,ln,m,n->Bijkl", bx1, by1, field, bx2, by2, wx, wy)
