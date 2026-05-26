# An implementation of Variationally Mimetic Operator Networks (VarMiONs)

import torch
from torch import nn
from ..symmetries import batch_expand, bspline_reduce_vectors
from ..subnets.basis import BSpline, TensorBasis, Functions
from ...utils.misc import einsum


class VarMiON(nn.Module):
    def __init__(
        self,
        problem: str,
        linears: nn.Module,
        branch: nn.Module,
        basis: nn.Module,
        config={"sga": "bspline", "scale": True},
        outputs="values",
    ):
        super().__init__()
        self.problem = problem
        self.linears = linears
        self.branch = branch
        self.basis = basis
        self.config = dict(config)
        self.outputs = outputs

    def forward(self, batch):
        assert (
            self.problem == batch["problem"]
        ), f"model is designed for {self.problem}, batch is from {batch['problem']}"
        branch_inputs, linear_inputs, scale = self.assemble(batch)
        mats = self.branch(branch_inputs)
        vecs = self.linears(linear_inputs)
        coeffs = einsum("Bij,Bj->Bi", mats, vecs)
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
        batch = dict(batch)
        has_v = batch["problem"] == "advection-diffusion"
        if self.config.get("sga") == "bspline":
            batch = batch_expand(batch)

        branch_fields = []
        if batch["problem"] == "diffusion" or batch["problem"] == "helmholtz":
            branch_fields = ["theta"]
        elif batch["problem"] == "advection-diffusion":
            branch_fields = ["theta", "v_x", "v_y"]

        # Flatten and concatenate fields
        scale = None
        if self.config.get("scale", False):
            # Scale theta (and v) so that theta is approximately 1. Because VarMIONs are linear,
            # we *could* scale the RHS terms here as well. However, for the use of a VarMiON as
            # preconditioner we have to also include the scaling in the resulting linear operator,
            # meaning we must "report" the scaling factor anyway.
            scale = 1 / batch["theta"].mean((1, 2))
            batch["theta"] = batch["theta"] * scale[:, None, None] - 1
            if has_v:
                batch["v_x"] = batch["v_x"] * scale[:, None, None]
                batch["v_y"] = batch["v_y"] * scale[:, None, None]

        branch_inputs = torch.cat([batch[key].flatten(1) for key in branch_fields], dim=1)
        linear_fields = ["f", "neumann:eta", "dirichlet:g"]
        linear_inputs = torch.cat([batch[key].flatten(1) for key in linear_fields], dim=1)

        return branch_inputs, linear_inputs, scale

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
            self.branch.to(torch.float64)
            self.linears.to(torch.complex128)
            return self
        else:
            return self.to(torch.float64)
