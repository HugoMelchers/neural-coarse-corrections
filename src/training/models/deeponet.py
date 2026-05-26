import torch
from torch import nn

from ..symmetries import batch_expand, bspline_reduce_vectors
from ..subnets.basis import BSpline, TensorBasis, Functions
from ...utils.misc import einsum


class DeepONet(nn.Module):
    def __init__(
        self,
        problem: str,
        branch: nn.Module,
        basis: nn.Module,
        config={"sga": "bspline", "scale": True},
        outputs="values",
    ):
        super().__init__()
        self.problem = problem
        self.branch = branch
        self.basis = basis
        self.config = dict(config)
        self.outputs = outputs

    def forward(self, batch):
        assert (
            self.problem == batch["problem"]
        ), f"model is designed for {self.problem}, batch is from {batch['problem']}"
        inputs, scale = self.assemble(batch)
        coeffs = self.branch(inputs)
        if scale is not None:
            coeffs = coeffs * scale[:, None]

        f_x = self.basis.f_x
        f_y = self.basis.f_y
        basis_x = f_x(batch["u/nodes_x"])
        basis_y = f_y(batch["u/nodes_y"])
        if self.config.get("sga") == "bspline":
            coeffs = coeffs.reshape(4, -1, len(basis_x), len(basis_y)).permute((1, 0, 2, 3))
            coeffs = bspline_reduce_vectors(coeffs)

        coeffs = coeffs.reshape(-1, len(basis_x), len(basis_y))
        d_basis_x = f_x.derivative(batch["u/nodes_x"])
        d_basis_y = f_y.derivative(batch["u/nodes_y"])
        basis_values = (basis_x, basis_y, d_basis_x, d_basis_y)
        return self.make_output(coeffs, scale, basis_values)

    def assemble(self, batch):
        # Make shallow copy. This is because we modify batch['theta'] but don't want to modify the
        # original batch dict.
        batch = dict(batch)
        has_v = self.problem == "advection-diffusion"
        is_helmholtz = self.problem == "helmholtz"
        if self.config.get("sga") == "bspline":
            batch = batch_expand(batch)

        if has_v:
            # For now, v is constant in space which means that for the parameter count we should
            # only include one sample of v instead of the same value 12² times
            batch["v_x"] = batch["v_x"][:, 0:1, 0:1]
            batch["v_y"] = batch["v_y"][:, 0:1, 0:1]

        fields = []
        if self.problem == "diffusion" or self.problem == "helmholtz":
            fields = ["theta", "f", "neumann:eta", "dirichlet:g"]
        elif self.problem == "advection-diffusion":
            fields = ["theta", "v_x", "v_y", "f", "neumann:eta", "dirichlet:g"]

        # Flatten and concatenate fields
        scale = None
        if self.config.get("scale", False):
            if not is_helmholtz:
                # Scale theta (and v) so that theta is approximately 1
                scale1 = 1 / batch["theta"].mean((1, 2))
                batch["theta"] = batch["theta"] * scale1[:, None, None] - 1
                if has_v:
                    batch["v_x"] = batch["v_x"] * scale1[:, None, None]
                    batch["v_y"] = batch["v_y"] * scale1[:, None, None]
            else:
                scale1 = 1

            # Scale f, g, and eta so that the maximum RMS of the three is 1
            f_rms = batch["f"].abs().square().mean((1, 2)).sqrt()
            g_rms = batch["dirichlet:g"].abs().square().mean((1, 2)).sqrt()
            eta_rms = batch["neumann:eta"].abs().square().mean((1, 2)).sqrt()
            scale2 = 1 / torch.maximum(f_rms, torch.maximum(g_rms, eta_rms))
            scale2 = torch.where(torch.isinf(scale2), 1, scale2)
            batch["f"] = batch["f"] * scale2[:, None, None]
            batch["dirichlet:g"] = batch["dirichlet:g"] * scale2[:, None, None]
            batch["neumann:eta"] = batch["neumann:eta"] * scale2[:, None, None]
            scale = scale1 / scale2

        inputs = torch.cat([batch[key].flatten(1) for key in fields], dim=1)

        return inputs, scale

    def make_output(self, coeffs, scale, basis_values):
        # Create a dict of the correct outputs
        # Options are: values, gradient, functions, bases
        outputs = self.outputs
        result = {}
        if isinstance(outputs, str):
            outputs = (outputs,)

        if "coeffs" in outputs:
            result["coeffs"] = coeffs

        functions = Functions(self.basis, coeffs)
        if "functions" in outputs:
            result["functions"] = functions

        basis_x, basis_y, d_basis_x, d_basis_y = basis_values
        if "values" in outputs:
            result["values"] = einsum("Bfg,fx,gy->Bxy", coeffs, basis_x, basis_y)

        if "gradient" in outputs:
            d_dx = einsum("Bfg,fx,gy->Bxy", coeffs, d_basis_x, basis_y)
            d_dy = einsum("Bfg,fx,gy->Bxy", coeffs, basis_x, d_basis_y)
            result["gradient"] = torch.stack((d_dx, d_dy), dim=1)

        if "bases" in outputs:
            result["basis_x"] = basis_x
            result["basis_y"] = basis_y

        # If self.outputs is a single string instead of a collection, also return a single output
        # instead of a dict
        if isinstance(self.outputs, str):
            result = result[self.outputs]

        return result

    def to_double(self):
        if self.problem == "helmholtz":
            self.basis.to(torch.float64)
            # self.branch.to(torch.complex128)
            self.branch = nn.Sequential(
                self.branch[0].to(torch.float64),
                *(layer.to(torch.complex128) for layer in self.branch[1:]),
            )
            return self
        else:
            return self.to(torch.float64)
