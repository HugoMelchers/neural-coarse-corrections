# 3D equivalents of `TensorGrid`, `TensorBasis`, and `Functions`:
# - TensorGrid3D: a tensor product of 1D nodes in x, y, and z
# - TensorBasis3D: a tensor product of three 1D bases
# - Functions3D: linear combinations of the functions in a 3D basis, with a fast path for evaluating
#   tensor bases on tensor grids
#
# Throughout, 3D arrays are ordered (x, y, z), so that when flattening them z varies fastest. This
# applies both to the points of a TensorGrid3D and to the functions of a TensorBasis3D, and is
# consistent with the 2D code (where y varies fastest).
#
# The interface is the same as in 2D:
# - f.forward(points) -> F × N
# - f.gradient(points) -> F × 3 × N (x-derivative, then y-derivative, then z-derivative)
# where points is either a N×3 array or a TensorGrid3D, and F = len(f) is the number of basis
# functions

import torch
from torch import nn

from src.training.subnets.basis import Functions
from src.utils.misc import einsum


class TensorGrid3D:
    def __init__(self, xs, ys=None, zs=None):
        """
        Construct a TensorGrid3D with the given x-, y-, and z-coordinates. If `ys` or `zs` are not
        given, `xs` is used instead.
        """
        self.xs = xs
        self.ys = ys if ys is not None else xs
        self.zs = zs if zs is not None else xs
        self.dtype = xs.dtype
        self.device = xs.device
        self.shape = (len(self.xs) * len(self.ys) * len(self.zs), 3)

    def full(self):
        """
        Return a `torch.Tensor` of shape (nx*ny*nz) × 3 containing all the points of this grid,
        ordered such that z varies fastest and x varies slowest.
        """
        return torch.cartesian_prod(self.xs, self.ys, self.zs)

    @property
    def grid_shape(self):
        return (self.xs.numel(), self.ys.numel(), self.zs.numel())

    def __str__(self):
        return "TensorGrid3D({}×{}×{})".format(*self.grid_shape)

    def __repr__(self):
        return f"TensorGrid3D({self.xs}, {self.ys}, {self.zs})"

    def to(self, device):
        return TensorGrid3D(self.xs.to(device), self.ys.to(device), self.zs.to(device))


class TensorBasis3D(nn.Module):
    dim = 3

    def __init__(self, f_x, f_y=None, f_z=None):
        """
        Create a 3D basis from a tensor product of 1D function bases `f_x`, `f_y`, `f_z`. If `f_y`
        or `f_z` are `None`, `f_x` is used along those axes instead.

        The basis functions are ordered with the z-index varying fastest, i.e.

            f_xyz(x, y, z)[(i * n_y + j) * n_z + k] = f_x[i](x) * f_y[j](y) * f_z[k](z)
        """
        super().__init__()
        self.f_x = f_x
        self.f_y = f_y or f_x
        self.f_z = f_z or f_x

    def _factors(self):
        return (self.f_x, self.f_y, self.f_z)

    def _values(self, points, derivs):
        # The 1D basis values (or derivatives) along each axis, each of size F_i × N_i
        if isinstance(points, TensorGrid3D):
            coords = (points.xs, points.ys, points.zs)
        else:
            coords = (points[:, 0], points[:, 1], points[:, 2])
        return [
            f(p) if d == 0 else f.derivative(p, d)
            for (f, p, d) in zip(self._factors(), coords, derivs)
        ]

    def _evaluate(self, points, derivs):
        a, b, c = self._values(points, derivs)
        if isinstance(points, TensorGrid3D):
            out = einsum("fx,gy,hz->fghxyz", a, b, c)
        else:
            out = einsum("fn,gn,hn->fghn", a, b, c)
        return out.reshape(len(a) * len(b) * len(c), -1)

    def forward(self, points):
        return self._evaluate(points, (0, 0, 0))

    def gradient(self, points):
        return torch.stack(
            [self._evaluate(points, derivs) for derivs in ((1, 0, 0), (0, 1, 0), (0, 0, 1))], 1
        )

    def gram_matrix(self):
        Mx = self.f_x.gram_matrix()
        My = self.f_y.gram_matrix()
        Mz = self.f_z.gram_matrix()
        return torch.kron(torch.kron(Mx, My), Mz)

    def inverse_gram_matrix(self):
        Mxi = self.f_x.inverse_gram_matrix()
        Myi = self.f_y.inverse_gram_matrix()
        Mzi = self.f_z.inverse_gram_matrix()
        return torch.kron(torch.kron(Mxi, Myi), Mzi)

    def __len__(self):
        return len(self.f_x) * len(self.f_y) * len(self.f_z)


def tensor_contract(coeffs, A, B, C):
    """
    Compute out[..., x, y, z] = Σ_{ijk} A[i, x] B[j, y] C[k, z] coeffs[..., i, j, k] one axis at a
    time, which is much cheaper than evaluating the full tensor product of A, B, and C first.
    """
    out = einsum("...ijk,ix->...xjk", coeffs, A)
    out = einsum("...xjk,jy->...xyk", out, B)
    return einsum("...xyk,kz->...xyz", out, C)


class Functions3D(Functions):
    """
    Same as `Functions`, but for 3D bases. When a `TensorBasis3D` is evaluated on a `TensorGrid3D`,
    the tensor product structure is used to evaluate the functions without forming the full
    (number of basis functions) × (number of points) matrix of the basis.
    """

    def _tensor_path(self, points):
        return isinstance(points, TensorGrid3D) and isinstance(self.basis, TensorBasis3D)

    def _evaluate(self, points, derivs):
        weights = self.get_weights_2d()
        basis = self.basis
        W = weights.reshape(-1, len(basis.f_x), len(basis.f_y), len(basis.f_z))
        a, b, c = basis._values(points, derivs)
        return tensor_contract(W, a, b, c).reshape(W.shape[0], -1)

    def forward(self, points):
        if self._tensor_path(points):
            return self.maybe_flatten(self._evaluate(points, (0, 0, 0)))
        return super().forward(points)

    def gradient(self, points):
        if self._tensor_path(points):
            output = torch.stack(
                [self._evaluate(points, derivs) for derivs in ((1, 0, 0), (0, 1, 0), (0, 0, 1))],
                dim=1,
            )
            return self.maybe_flatten(output)
        return super().gradient(points)

    def __getitem__(self, ids):
        return Functions3D(self.basis, self.weights[ids])
