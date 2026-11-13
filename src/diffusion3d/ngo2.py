import torch

from src.training.models.ngo import NGO
from src.training.subnets.basis import BSpline
from src.utils.misc import einsum
from .basis import TensorBasis3D, Functions3D, tensor_contract
from .symmetries import (
    bspline_expand_vectors_3d,
    bspline_reduce_vectors_3d,
    bspline_reduce_matrices_3d,
)


def _integrate(F, A, B, C):
    """
    Given a batch of samples F (B × nx × ny × nz) and test functions sampled on the quadrature
    nodes and multiplied by the quadrature weights (A: Fx × nx, B: Fy × ny, C: Fz × nz), compute
    out[B, i, j, k] = Σ_{xyz} A[i, x] B[j, y] C[k, z] F[B, x, y, z].
    """
    return tensor_contract(F, A.T, B.T, C.T)


def data_ngo_assemble_diffusion_3d(batch, basis, config):
    """
        data_ngo_assemble_diffusion_3d(batch, basis, config)

    3D equivalent of `data_ngo_assemble_diffusion`: assemble a batch of 3D diffusion problems for a
    data NGO.

    - `batch` should be a dict with at least the following keys:
        - `nodes_x`, `weights_x`, `nodes_y`, `weights_y`, `nodes_z`, `weights_z`: quadrature rules
          along each axis, with nx, ny, and nz nodes, respectively
        - `theta`, `f`: arrays of size B × nx × ny × nz containing the diffusivity and source term
        - `dirichlet:g`: array of size B × 2 × ny × nz containing the Dirichlet data on x = 0, 1
        - `neumann_y:eta`: array of size B × nx × 2 × nz containing the Neumann data on y = 0, 1
        - `neumann_z:eta`: array of size B × nx × ny × 2 containing the Neumann data on z = 0, 1
    - `basis` should be a TensorBasis3D with BSpline bases for f_x, f_y, and f_z
    - `config`: a `dict` with the same (optional) keys as in 2D: `sga`, `scale`, and `gamma`

    The output is a triple `K, b, scale` where `K` and `b` are both B × 1 × Nx × Ny × Nz, with Nx,
    Ny, and Nz the number of basis functions along each axis.
    """
    assert isinstance(basis, TensorBasis3D)
    assert all(isinstance(f, BSpline) for f in (basis.f_x, basis.f_y, basis.f_z))
    sga = config.get("sga", None)
    gamma = config.get("gamma", None)
    assert sga in [None, "bspline"]

    weights_x, weights_y, weights_z = batch["weights_x"], batch["weights_y"], batch["weights_z"]
    dtype, device = weights_x.dtype, weights_x.device
    f_x, f_y, f_z = basis.f_x, basis.f_y, basis.f_z
    # Test functions multiplied by the quadrature weights
    v_x = f_x(batch["nodes_x"]) * weights_x
    v_y = f_y(batch["nodes_y"]) * weights_y
    v_z = f_z(batch["nodes_z"]) * weights_z
    # On the boundaries, the "quadrature" consists of a single node with weight 1
    nodes_b = torch.tensor([0.0, 1.0], dtype=dtype, device=device)
    normals = torch.tensor([-1.0, 1.0], dtype=dtype, device=device)
    v_x_b = f_x(nodes_b)
    v_y_b = f_y(nodes_b)
    v_z_b = f_z(nodes_b)
    dv_x_dn = f_x.derivative(nodes_b) * normals

    theta = batch["theta"]
    f = batch["f"]
    g = batch["dirichlet:g"]
    eta_y = batch["neumann_y:eta"]
    eta_z = batch["neumann_z:eta"]

    scale = config.get("scale", False)
    scale_factor = None
    if scale:
        # Scale theta such that its average equals 1
        # Then integrate theta - 1 against the basis, and also scale the right-hand-side vector
        theta_avg = torch.einsum("Bxyz,x,y,z->B", theta, weights_x, weights_y, weights_z)
        scale_factor = 1 / theta_avg
        theta = theta * scale_factor[:, None, None, None] - 1

    # Matrix input: ⟨v, θ⟩_Ω
    K = _integrate(theta, v_x, v_y, v_z)

    # Vector input: ⟨v, f⟩_Ω - ⟨∇v⋅n⃗, g⟩_ΓD + γ⟨v, g⟩_ΓD + ⟨v, η⟩_ΓN
    # Here γ is the Nitsche stabilisation coefficient, which may be zero
    b = _integrate(f, v_x, v_y, v_z)
    b = b - _integrate(g, dv_x_dn, v_y, v_z)
    if gamma is not None:
        b = b + gamma * _integrate(g, v_x_b, v_y, v_z)

    b = b + _integrate(eta_y, v_x, v_y_b, v_z)
    b = b + _integrate(eta_z, v_x, v_y, v_z_b)

    # Add a singleton index since we have one channel in both K and b
    return K[:, None], b[:, None], scale_factor


class NGO_3D(NGO):
    def __init__(
        self,
        *,
        problem: str = "diffusion3d",
        system_net,
        basis: TensorBasis3D,
        config={"sga": "bspline", "scale": True},
        outputs="values",
    ):
        """
        3D equivalent of `NGO`. Arguments:
        - problem: the problem this network is trained on. Currently only "diffusion3d"
        - system_net: a network that maps inputs of shape B × 1 × nx × ny × nz (the assembled
          diffusivity) to matrices of shape B × N × N with N = nx * ny * nz
        - basis: the 3D basis functions, a TensorBasis3D of B-splines
        - config: a dictionary containing other options. By default it enables symmetry group
          averaging (sga = "bspline") and scaling invariance (scale = True)
        - outputs: either a string or a collection of strings representing what should be returned.
          The options are the same as for `NGO`: values, gradient, functions, coeffs, bases,
          matrices
        """
        super().__init__(
            problem=problem, system_net=system_net, basis=basis, config=config, outputs=outputs
        )

    def _factors(self):
        return (self.basis.f_x, self.basis.f_y, self.basis.f_z)

    def _basis_values(self, nodes):
        # Returns (basis_x, basis_y, basis_z, d_basis_x, d_basis_y, d_basis_z)
        values = tuple(f(n) for (f, n) in zip(self._factors(), nodes))
        derivatives = tuple(f.derivative(n) for (f, n) in zip(self._factors(), nodes))
        return values + derivatives

    def forward(self, batch):
        assert (
            self.problem == batch["problem"]
        ), f"model is designed for {self.problem}, batch is from {batch['problem']}"
        K, b, scale = self.assemble(batch)
        nodes = (batch["u/nodes_x"], batch["u/nodes_y"], batch["u/nodes_z"])
        coeffs, matrices = self.perform_matmul(K, b)
        return self.make_output(coeffs, matrices, scale, self._basis_values(nodes))

    def preassemble_attributes(self, attributes):
        nodes = (attributes["u/nodes_x"], attributes["u/nodes_y"], attributes["u/nodes_z"])
        values = self._basis_values(nodes)
        names = ("basis_x", "basis_y", "basis_z", "d_basis_x", "d_basis_y", "d_basis_z")
        return {
            "u/weights_x": attributes["u/weights_x"],
            "u/weights_y": attributes["u/weights_y"],
            "u/weights_z": attributes["u/weights_z"],
            **dict(zip(names, values)),
        }

    def call_system_net(self, batch):
        K = batch["input_mat"]
        b = batch["input_vec"]
        scale = batch.get("scale")
        names = ("basis_x", "basis_y", "basis_z", "d_basis_x", "d_basis_y", "d_basis_z")
        basis_values = tuple(batch[name] for name in names)
        coeffs, matrices = self.perform_matmul(K, b)
        return self.make_output(coeffs, matrices, scale, basis_values)

    def make_output(self, coeffs, matrices, scale, basis_values):
        # Create a dict of the correct outputs
        # Options are: values, gradient, functions, coeffs, bases, matrices
        outputs = self.outputs
        result = {}
        if isinstance(outputs, str):
            outputs = (outputs,)

        if "matrices" in outputs:
            if scale is not None:
                matrices = matrices * scale[:, None, None]
            result["matrices"] = matrices

        if scale is not None:
            coeffs = coeffs * scale[:, None, None, None]

        if "functions" in outputs:
            result["functions"] = Functions3D(self.basis, coeffs.flatten(1))

        if "coeffs" in outputs:
            result["coeffs"] = coeffs

        basis_x, basis_y, basis_z, d_basis_x, d_basis_y, d_basis_z = basis_values
        if "values" in outputs:
            result["values"] = tensor_contract(coeffs, basis_x, basis_y, basis_z)

        if "gradient" in outputs:
            d_dx = tensor_contract(coeffs, d_basis_x, basis_y, basis_z)
            d_dy = tensor_contract(coeffs, basis_x, d_basis_y, basis_z)
            d_dz = tensor_contract(coeffs, basis_x, basis_y, d_basis_z)
            result["gradient"] = torch.stack((d_dx, d_dy, d_dz), dim=1)

        if "bases" in outputs:
            result["basis_x"] = basis_x
            result["basis_y"] = basis_y
            result["basis_z"] = basis_z

        # If self.outputs is a single string instead of a collection, also return a single output
        # instead of a dict
        if isinstance(self.outputs, str):
            result = result[self.outputs]

        return result

    def assemble(self, batch):
        if batch["problem"] == "diffusion3d":
            return data_ngo_assemble_diffusion_3d(batch, self.basis, self.config)
        else:
            raise ValueError(f"unsupported PDE \"{batch['problem']}\"")

    def perform_matmul(self, K, b):
        # Input sizes: K and b are both B × 1 × nx × ny × nz
        err = f"NGO_3D: invalid input shapes {K.shape}, {b.shape}"
        assert K.ndim == 5 and b.ndim == 5, err
        assert K.shape == b.shape and K.shape[1] == 1, err

        B, _, nx, ny, nz = b.shape
        N = nx * ny * nz
        sga = self.config.get("sga")
        if sga == "bspline":
            # Expand to the 8 mirrored versions of each problem, which become a batch of size 8B
            b = bspline_expand_vectors_3d(b[:, 0]).flatten(0, 1)[:, None]
            K = bspline_expand_vectors_3d(K[:, 0]).flatten(0, 1)[:, None]

        matrices = self.system_net(K)
        coeffs = einsum("Bij,Bj->Bi", matrices, b.flatten(1))
        coeffs = coeffs.reshape(-1, nx, ny, nz)

        if sga == "bspline":
            # Reduce
            coeffs = bspline_reduce_vectors_3d(coeffs.reshape(B, 8, nx, ny, nz))
            matrices = matrices.reshape(B, 8, nx, ny, nz, nx, ny, nz)
            matrices = bspline_reduce_matrices_3d(matrices).reshape(B, N, N)

        return coeffs, matrices
