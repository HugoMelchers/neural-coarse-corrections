import torch

from src.training.subnets import TensorBasis, BSpline


def data_ngo_assemble_diffusion(batch, basis, config):
    """
        data_ngo_assemble_diffusion(batch, basis, config)

    Assemble a batch of diffusion problems for a data NGO.

    - `batch` should be a dict with at least the following keys:
        - `nodes_x`, `weights_x`: quadrature rule along the x-axis, both vectors with length nx
        - `nodes_y`, `weights_y`: quadrature rule along the y-axis, both vectors with length ny
        - `theta`, `f`: arrays of size B × nx × ny containing the diffusivity and source term
        - `dirichlet:g`: array of size B × 2 × ny containing the Dirichlet boundary data
        - `neumann:eta`: array of size B × nx × 2 containing the Neumann boundary data
    - `basis` should be a TensorBasis with BSpline bases for f_x and f_y
    - `config`: a `dict` with the following keys, all optional:
        - `sga`: symmetry group averaging. `None` (default) or "bspline"
        - `scale`: scale equivariance. `False` (default) or `True`
        - `gamma`: weak Dirichlet imposition stabilisation coefficient. `None` (default) or float

    The output will be a pair `K, b` where `K` and `b` are both B × Nx × Ny with Nx and Ny the
    number of basis functions in the x amd y directions, respectively.
    """
    assert isinstance(basis, TensorBasis)
    assert isinstance(basis.f_x, BSpline)
    assert isinstance(basis.f_y, BSpline)
    sga = config.get("sga", None)
    gamma = config.get("gamma", None)
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

    scale = config.get("scale", False)
    scale_factor = None
    if scale:
        # Scale theta such that its average equals 1
        # Then integrate theta - 1 against the basis, and also scale the right-hand-side vector
        theta_avg = torch.einsum("Bxy,x,y->B", theta, weights_x, weights_y)
        scale_factor = 1 / theta_avg
        theta = theta * scale_factor[:, None, None] - 1

    # Matrix input: ⟨v, θ⟩_Ω
    K = torch.einsum("ix,x,Bxy,y,jy->Bij", basis_x, weights_x, theta, weights_y, basis_y)

    # Vector input: ⟨v, f⟩_Ω - ⟨∇v⋅n⃗, g⟩_ΓD + γ⟨v, g⟩_ΓD + ⟨v, η⟩_ΓN
    # Here γ is the Nitsche stabilisation coefficient, which may be zero
    b = torch.einsum("ix,x,Bxy,y,jy->Bij", basis_x, weights_x, f, weights_y, basis_y)
    b = b - torch.einsum(
        "ix,x,x,Bxy,y,jy->Bij", d_basis_x_b, weights_b, normals, g, weights_y, basis_y
    )
    if gamma is not None:
        b = b + gamma * torch.einsum(
            "ix,x,Bxy,y,jy->Bij", basis_x_b, weights_b, g, weights_y, basis_y
        )

    b = b + torch.einsum("ix,x,Bxy,y,jy->Bij", basis_x, weights_x, eta, weights_b, basis_y_b)

    # Add a singleton index since we have one channel in both K and b
    return K[:, None], b[:, None], scale_factor
