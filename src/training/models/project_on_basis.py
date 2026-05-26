import torch
from torch import nn
from src.utils.misc import einsum


class ProjectOnBasis(nn.Module):
    def __init__(self, basis, config={}, outputs="values"):
        """
        Create a model that simply projects the true input solution (batch["u"]) onto the span of
        the basis. This is only really useful to train trunk nets, or to use as a reference to see
        whether the trunk is limiting the accuracy of a model.
        """
        super().__init__()
        self.basis = basis
        self.config = config
        self.outputs = outputs

    def forward(self, batch):
        true_solution = batch["u"]
        f_x = self.basis.f_x
        f_y = self.basis.f_y
        basis_x = f_x(batch["u/nodes_x"])
        basis_y = f_y(batch["u/nodes_y"])
        weights_x = batch["u/weights_x"]
        weights_y = batch["u/weights_y"]
        Mx = torch.einsum("ix,jx,x->ij", basis_x, basis_x, weights_x)
        My = torch.einsum("ix,jx,x->ij", basis_y, basis_y, weights_y)
        Mxi = Mx.inverse()
        Myi = My.inverse()
        inp = einsum("Bxy,ix,jy,x,y->Bij", true_solution, basis_x, basis_y, weights_x, weights_y)
        coeffs = einsum("ij,Bjk,kl->Bil", Mxi, inp, Myi)
        values = einsum("ix,Bij,jy->Bxy", basis_x, coeffs, basis_y)
        return values
