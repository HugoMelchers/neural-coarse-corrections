import torch

from src.training.subnets import TensorBasis, BSpline
from src.utils.misc import einsum


def data_ngo_assemble_helmholtz(batch, basis, config):
    """
        data_ngo_assemble_helmholtz(batch, basis, config)

    Assemble a batch of Helmholtz problems for a data NGO.

    - `batch` should be a dict with at least the following keys:
        - `nodes_x`, `weights_x`: quadrature rule along the x-axis, both vectors with length nx
        - `nodes_y`, `weights_y`: quadrature rule along the y-axis, both vectors with length ny
        - `theta`, `f`: arrays of size B × nx × ny containing the diffusivity and source term
        - `dirichlet:g`: array of size B × 2 × ny containing the left/right Dirichlet boundary data
        - `neumann:eta`: array of size B × nx × 2 containing the bottom/top Dirichlet boundary data
            - Note that the naming is copied from the other test cases
    - `basis` should be a TensorBasis with BSpline bases for f_x and f_y
    - `config`: a `dict` with the following keys, all optional:
        - `sga`: symmetry group averaging. `None` (default) or "bspline"

    The output will be a pair `K, b` where `K` and `b` are both B × Nx × Ny with Nx and Ny the
    number of basis functions in the x amd y directions, respectively.
    """
    assert isinstance(basis, TensorBasis)
    assert isinstance(basis.f_x, BSpline)
    assert isinstance(basis.f_y, BSpline)
    sga = config.get("sga", None)
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
    g_left_right = batch["dirichlet:g"]
    g_bottom_top = batch["neumann:eta"]

    scale_factor = None

    # Matrix input: ⟨v, θ⟩_Ω
    K = torch.einsum("ix,x,Bxy,y,jy->Bij", basis_x, weights_x, theta, weights_y, basis_y)

    # Vector input: ⟨v, f⟩_Ω - ⟨∇v⋅n⃗, g⟩_ΓD
    b = einsum("ix,x,Bxy,y,jy->Bij", basis_x, weights_x, f, weights_y, basis_y)
    # Left/right boundary
    b = b - einsum(
        "ix,x,x,Bxy,y,jy->Bij", d_basis_x_b, weights_b, normals, g_left_right, weights_y, basis_y
    )
    # Bottom/top boundary
    b = b - einsum(
        "ix,x,Bxy,y,y,jy->Bij", basis_x, weights_x, g_bottom_top, normals, weights_b, d_basis_y_b
    )

    # Add a singleton index since we have one channel in both K and b
    return K[:, None], b[:, None], None
