# An implementation of a Resolution Independent Neural Operator (RINO) for diffusion. See Bahmani
# et al., A resolution independent neural operator (DOI: https://doi.org/10.1016/j.cma.2025.118113).
# Note that this is really a RI-DeepONet and not a RINO, but the two only differ in their loss
# function.

import torch
from torch import nn
from .. import symmetries
from ..subnets.basis import BSpline, TensorBasis, Functions
from ...utils.misc import einsum


class RINO(nn.Module):
    def __init__(
        self,
        problem: str,
        branch,
        basis,
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
        sga = self.config.get("sga") == "bspline"
        batch, scale = RINO._scalebatch(batch) if self.config.get("scale", False) else (batch, None)
        coeffs = self.project_fields(batch)
        if self.problem == "advection-diffusion":
            coeffs = coeffs[:-2] + [coeffs[-2][:, 0:1, 0:1]] + [coeffs[-1][:, 0:1, 0:1]]
        if sga:
            # Currently hard-coded to work on a 12x12 basis
            if self.problem == "advection-diffusion":
                coeffs = (
                    [symmetries.bspline_expand_vectors(c) for c in coeffs[:-2]]
                    + [symmetries.bspline_expand_vectors(coeffs[-2], "x")]
                    + [symmetries.bspline_expand_vectors(coeffs[-1], "y")]
                )
            else:
                coeffs = [symmetries.bspline_expand_vectors(c) for c in coeffs]

            coeffs = [c.flatten(0, 1) for c in coeffs]

        inputs = torch.cat([c.flatten(1) for c in coeffs], dim=1)
        nx, ny = len(self.basis.f_x), len(self.basis.f_y)
        coeffs_u = self.branch(inputs).reshape(-1, nx, ny)
        if sga:
            coeffs_u = symmetries.bspline_reduce_vectors(coeffs_u.reshape(-1, 4, nx, ny))

        if scale is not None:
            coeffs_u = coeffs_u * scale[:, None, None]
        basis_x = self.basis.f_x(batch["u/nodes_x"])
        basis_y = self.basis.f_y(batch["u/nodes_y"])
        d_basis_x = self.basis.f_x.derivative(batch["u/nodes_x"])
        d_basis_y = self.basis.f_y.derivative(batch["u/nodes_y"])
        return self.make_output(coeffs_u, (basis_x, basis_y, d_basis_x, d_basis_y))

    def project_fields(self, batch):
        dtype, device = batch["nodes_x"].dtype, batch["nodes_y"].device
        n01 = torch.tensor([0, 1], dtype=dtype, device=device)
        w01 = torch.tensor([1, 1], dtype=dtype, device=device)
        c_g = RINO._project(
            self.basis,
            (n01, batch["left:g/nodes"]),
            (w01, batch["left:g/weights"]),
            [batch["dirichlet:g"]],
            self.config,
        )[0][:, [0, -1], :]
        c_eta = RINO._project(
            self.basis,
            (batch["bottom:eta/nodes"], n01),
            (batch["bottom:eta/weights"], w01),
            [batch["neumann:eta"]],
            self.config,
        )[0][:, :, [0, -1]]

        if self.problem == "advection-diffusion":
            c_theta, c_vx, c_vy, c_f = RINO._project(
                self.basis,
                (batch["u/nodes_x"], batch["u/nodes_y"]),
                (batch["u/weights_x"], batch["u/weights_y"]),
                [batch["theta"], batch["v_x"], batch["v_y"], batch["f"]],
                self.config,
            )
            return [c_theta, c_f, c_eta, c_g, c_vx, c_vy]
        else:
            c_theta, c_f = RINO._project(
                self.basis,
                (batch["u/nodes_x"], batch["u/nodes_y"]),
                (batch["u/weights_x"], batch["u/weights_y"]),
                [batch["theta"], batch["f"]],
                self.config,
            )
            return [c_theta, c_f, c_eta, c_g]

    def make_output(self, coeffs, basis_values):
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
                *(layer.to(torch.complex128) for layer in self.branch[1:])
            )
            return self
        else:
            return self.to(torch.float64)

    @staticmethod
    def _get_M_1d_inv(basis, nodes, weights):
        values = basis(nodes)
        if nodes.numel() == 2:
            # This is a boundary quadrature rule, so use the pseudo-inverse
            dtype, device = nodes.dtype, nodes.device
            k = len(basis)
            Mi = torch.zeros((k, k), dtype=dtype, device=device)
            Mi[0, 0] = 1
            Mi[-1, -1] = 1
        else:
            Mi = basis.inverse_gram_matrix()

        return values, Mi

    @staticmethod
    def _get_M_inv(basis, nodes, weights):
        vx, Mx_inv = RINO._get_M_1d_inv(basis.f_x, nodes[0], weights[0])
        vy, My_inv = RINO._get_M_1d_inv(basis.f_y, nodes[1], weights[1])
        return vx, vy, Mx_inv, My_inv

    @staticmethod
    def _project(basis, nodes, weights, fields, config):
        vx, vy, Mx_i, My_i = RINO._get_M_inv(basis, nodes, weights)
        wx, wy = weights
        rhss = [einsum("ix,x,Bxy,y,jy->Bij", vx, wx, field, wy, vy) for field in fields]
        if config.get("no-M-inv", False):
            # If configured: don't project the fields onto the basis, but just integrate them
            # against the basis. This is more similar to the NGO, and, depending on the training
            # data, may also be more well-conditioned thereby speeding up convergence during
            # training.
            return rhss
        else:
            return [einsum("ix,Bxy,yj", Mx_i, rhs, My_i) for rhs in rhss]

    @staticmethod
    def _scalebatch(batch):
        batch = dict(batch)
        is_helmholtz = batch["problem"] == "helmholtz"
        avg_theta = einsum(
            "Bxy,x,y->B", batch["theta"], batch["theta/weights_x"], batch["theta/weights_y"]
        )
        if not is_helmholtz:
            scale1 = 1 / avg_theta
            batch["theta"] = batch["theta"] * scale1[:, None, None] - 1
            if batch["problem"] == "advection-diffusion":
                batch["v_x"] = batch["v_x"] * scale1[:, None, None]
                batch["v_y"] = batch["v_y"] * scale1[:, None, None]
        else:
            scale1 = 1

        rms_f = einsum(
            "Bxy,x,y->B", batch["f"].abs().square(), batch["f/weights_x"], batch["f/weights_y"]
        ).sqrt()
        rms_eta = einsum(
            "Bxy,x->B", batch["neumann:eta"].abs().square(), batch["bottom:eta/weights"] / 2
        ).sqrt()
        rms_g = einsum(
            "Bxy,y->B", batch["dirichlet:g"].abs().square(), batch["left:g/weights"] / 2
        ).sqrt()

        scale2 = 1 / torch.maximum(rms_f, torch.maximum(rms_eta, rms_g))
        scale2 = torch.where(torch.isinf(scale2), 1, scale2)
        batch["f"] = batch["f"] * scale2[:, None, None]
        batch["dirichlet:g"] = batch["dirichlet:g"] * scale2[:, None, None]
        batch["neumann:eta"] = batch["neumann:eta"] * scale2[:, None, None]
        return batch, scale1 / scale2
