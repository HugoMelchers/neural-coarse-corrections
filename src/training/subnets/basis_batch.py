# NGOs with problem-dependent basis often don't work with a single basis, but a batch of function
# bases with one for each problem. Right now, the only type of batched basis is a batch of BSplines,
# so this code is specifically for that kind of basis.
import torch
from src.training.subnets.basis import Functions


class BSplineBatch(torch.nn.Module):
    def __init__(self, knots, degree):
        """
        Construct a 1D B-spline Basis Batch
        """
        super().__init__()
        self.knots = knots
        self.degree = degree
        self.dim = 1

    def forward(self, xs):
        return _bspline_get_values_batch(self.knots, self.degree, xs)

    def derivative(self, xs, k=1):
        return _bspline_get_values_batch(self.knots, self.degree, xs, deriv=k)


def _bspline_get_values_batch(knots, degree, x, deriv=0):
    """
    Get the values of a batch of B-spline bases on the nodes x.

    - knots: a B × k array of B different knot vectors
    - degree: an integer, the degree of the B-spline bases (the same for all bases in the batch)
    - x: an array of n points to evaluate the bases on
    - deriv: an integer indicating the number of derivatives to take. Default and minimum is 0,
      should be smaller than or equal to `degree`.

    Output: an array of size B × (k - deg) × n array, indexed by batch / function / point
    """
    assert knots.ndim == 2
    assert x.ndim == 1
    B, nk = knots.shape
    Nx = x.numel()
    # Add singleton dimensions to the knots and input points in the correct places
    knots = knots[:, :, None]
    x = x[None, None, :]
    dtype, device = x.dtype, x.device
    padding = torch.full((B, degree, Nx), False, device=x.device)
    result = torch.cat(
        (
            padding,
            x < knots[:, degree + 1 : degree + 2],
            (knots[:, degree + 1 : -degree - 2] <= x) * (x < knots[:, degree + 2 : -degree - 1]),
            knots[:, -degree - 2 : -degree - 1] <= x,
            padding,
        ),
        dim=1,
    ).to(x.dtype)
    eps = torch.tensor(1e-10, dtype=x.dtype, device=x.device)
    for deg in range(1, degree + 1):
        nf = nk - deg - 1
        a = knots[:, 0 : nf + 1]
        b = knots[:, deg : (1 + deg + nf)]
        l = torch.maximum(b - a, eps)
        if degree - deg >= deriv:
            v1 = result[:, 0:nf] * (x - a[:, :-1]) / l[:, :-1]
            v2 = result[:, 1 : (nf + 1)] * (b[:, 1:] - x) / l[:, 1:]
            result = v1 + v2
        else:
            # Differentiation step
            # α_view .= p ./ getindex.(Ref(diffs), js)
            w = deg / l
            result = w[:, 0:nf] * result[:, 0:nf] - w[:, 1 : (nf + 1)] * result[:, 1 : (nf + 1)]

    return result
