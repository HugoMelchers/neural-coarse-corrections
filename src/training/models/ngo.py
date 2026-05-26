from typing import Union

import torch
from torch import nn

from ..symmetries import (
    bspline_expand_vectors,
    bspline_reduce_vectors,
    bspline_expand_matrices,
    bspline_reduce_matrices,
)
from ..subnets.basis import BSpline, TensorBasis, Functions
from .ngo_assembly import (
    data_ngo_assemble_diffusion,
    data_ngo_assemble_advdiff,
    data_ngo_assemble_helmholtz,
)
from src.utils.misc import einsum


class NGO(nn.Module):
    def __init__(
        self,
        *,
        problem: str,
        system_net: nn.Module,
        basis: nn.Module,
        config={"sga": "bspline", "scale": True},
        outputs="values",
    ):
        """
        Construct an NGO. Arguments:
        - problem: the problem this network is trained on, i.e. "diffusion" or "advection-diffusion"
        - system_net: a SemiLinearUNet.
        - basis: the 2D basis functions. These are expected to be a tensor basis
        - config: a dictionary containing other options. By default it enables symmetry group
          averaging (sga = True) and scaling invariance (scale = True)
        - outputs: either a string or a collection of strings representing what should be returned
        """
        super().__init__()
        self.problem = problem
        self.system_net = system_net
        self.basis = basis
        self.config = dict(config)
        self.outputs = outputs

    def forward(self, batch):
        assert (
            self.problem == batch["problem"]
        ), f"model is designed for {self.problem}, batch is from {batch['problem']}"
        K, b, scale = self.assemble(batch)
        f_x = self.basis.f_x
        f_y = self.basis.f_y
        basis_x = f_x(batch["u/nodes_x"])
        basis_y = f_y(batch["u/nodes_y"])
        d_basis_x = f_x.derivative(batch["u/nodes_x"])
        d_basis_y = f_y.derivative(batch["u/nodes_y"])
        basis_values = (basis_x, basis_y, d_basis_x, d_basis_y)
        coeffs, matrices = self.perform_matmul(K, b)
        coeffs = coeffs.reshape(-1, len(basis_x), len(basis_y))
        return self.make_output(coeffs, matrices, scale, basis_values)

    def preassemble_batch(self, batch):
        assert (
            self.problem == batch["problem"]
        ), f"model is designed for {self.problem}, batch is from {batch['problem']}"
        K, b, scale = self.assemble(batch)
        output = {
            "input_mat": K,
            "input_vec": b,
            "u": batch["u"],
        }
        if scale is not None:
            output["scale"] = scale

        return output

    def preassemble_attributes(self, attributes):
        f_x = self.basis.f_x
        f_y = self.basis.f_y
        return {
            "u/weights_x": attributes["u/weights_x"],
            "u/weights_y": attributes["u/weights_y"],
            "basis_x": f_x(attributes["u/nodes_x"]),
            "basis_y": f_y(attributes["u/nodes_y"]),
            "d_basis_x": f_x.derivative(attributes["u/nodes_x"]),
            "d_basis_y": f_y.derivative(attributes["u/nodes_y"]),
        }

    def call_system_net(self, batch):
        K = batch["input_mat"]
        b = batch["input_vec"]
        scale = batch.get("scale")
        basis_x = batch["basis_x"]
        basis_y = batch["basis_y"]
        d_basis_x = batch["d_basis_x"]
        d_basis_y = batch["d_basis_y"]
        coeffs, matrices = self.perform_matmul(K, b)
        coeffs = coeffs.reshape(-1, len(basis_x), len(basis_y))
        basis_values = (basis_x, basis_y, d_basis_x, d_basis_y)
        return self.make_output(coeffs, matrices, scale, basis_values)

    def make_output(self, coeffs, matrices, scale, basis_values):
        # Create a dict of the correct outputs
        # Options are: values, gradient, functions, bases, matrices
        outputs = self.outputs
        result = {}
        if isinstance(outputs, str):
            outputs = (outputs,)

        if "matrices" in outputs:
            if scale is not None:
                matrices = matrices * scale[:, None, None]
            result["matrices"] = matrices

        if scale is not None:
            coeffs = coeffs * scale[:, None, None]

        functions = Functions(self.basis, coeffs)
        if "functions" in outputs:
            result["functions"] = functions

        if "coeffs" in outputs:
            result["coeffs"] = coeffs

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

    def assemble(self, batch):
        if batch["problem"] == "diffusion":
            return data_ngo_assemble_diffusion(batch, self.basis, self.config)
        elif batch["problem"] == "advection-diffusion":
            return data_ngo_assemble_advdiff(batch, self.basis, self.config)
        elif batch["problem"] == "helmholtz":
            return data_ngo_assemble_helmholtz(batch, self.basis, self.config)
        else:
            raise ValueError(f"unsupported PDE \"{batch['problem']}\"")

    def perform_matmul(self, K, b):
        # Input sizes:
        # - Data NGO: K is B × c1 × nx × ny, b is B × c2 × nx × ny
        #   B: batch size
        #   c1: number of channels in K = number of PDE coefficients (diffusivity / velocity / ...)
        #   c2: number of channels in b = number of source terms
        #   nx: number of basis functions in x-axis
        #   ny: number of basis functions in y-axis
        err = f"NGO: invalid input shapes {K.shape}, {b.shape}"
        assert K.ndim == 4, err
        assert b.ndim == 4, err
        cK = 1 if self.problem in ["diffusion", "helmholtz"] else 3
        cb = 1
        B, cK1, nx, ny = K.shape
        B1, cb1, nx1, ny1 = b.shape
        assert B == B1, err
        assert cK == cK1, err
        assert cb == cb1, err
        assert nx == nx1, err
        assert ny == ny1, err

        B = b.shape[0]
        nx, ny = b.shape[-2:]
        sga = self.config.get("sga")
        B = K.shape[0]
        if sga == "bspline":
            # Expand. This always the same for model NGOs, but for data NGOs it depends on the PDE
            # Note that the bspline_expand_* functions produce arrays with shape B × 4 × ..., but
            # for the system network these have to be reshaped to 4B × ...
            # b has a single channel, so do symmetry expansions on that
            # Note that array[:, 0] removes a singleton dimension at index 1, and array[:, None]
            # inserts one
            b = bspline_expand_vectors(b[:, 0]).flatten(0, 1)[:, None]
            if self.problem in ["diffusion", "helmholtz"]:
                K = bspline_expand_vectors(K[:, 0]).flatten(0, 1)[:, None]
            elif self.problem == "advection-diffusion":
                K1, K2, K3 = K[:, 0], K[:, 1], K[:, 2]
                K1 = bspline_expand_vectors(K1).flatten(0, 1)
                K2 = bspline_expand_vectors(K2, negate="x").flatten(0, 1)
                K3 = bspline_expand_vectors(K3, negate="y").flatten(0, 1)
                K = torch.stack((K1, K2, K3), dim=1)

        matrices = self.system_net(K)
        coeffs = einsum("Bij,Bj->Bi", matrices, b.flatten(1))
        coeffs = coeffs.reshape(-1, nx, ny)

        if sga == "bspline":
            # Reduce
            coeffs = coeffs.reshape(B, 4, *coeffs.shape[1:])
            coeffs = bspline_reduce_vectors(coeffs)
            matrices = bspline_reduce_matrices(matrices.reshape(B, 4, nx, ny, nx, ny)).reshape(
                B, nx * ny, nx * ny
            )

        return coeffs, matrices

    def to_double(self):
        return self.to(torch.float64)
