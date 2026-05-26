import numpy as np
import torch
from typing import Tuple, Union
import re


class TensorGrid:
    def __init__(self, xs, ys=None, x_varies_first=False):
        """
        Construct a TensorGrid with the given x-coordinates and y-coordinates.
        """
        self.xs = xs
        self.ys = ys if ys is not None else xs
        self.x_varies_first = x_varies_first
        self.dtype = xs.dtype
        self.device = xs.device
        self.shape = (len(xs) * len(self.ys), 2)

    def full(self):
        """
        Return a `torch.Tensor` containing all the points represented by this grid. The ordering of
        the points depends on `self.x_varies_first` (defaults to False). If this grid has
        `len(xs) == m` and `len(ys) == n`, then the output will have shape `(m*n) × 2`, where

        x_varies_first == False: XY[n * i + j, :] = [xs[i], ys[j]]
        x_varies_first == True : XY[i + m * j, :] = [xs[i], ys[j]]
        """
        if self.x_varies_first:
            return torch.cartesian_prod(self.ys, self.xs).flip(1)
        else:
            return torch.cartesian_prod(self.xs, self.ys)

    @staticmethod
    def from_tensor(XY):
        """
        If `XY` is a list of points constructed from a tensor grid, turn it back into a `TensorGrid`
        representation. Note that this function always checks that output.full() == XY, and fails
        with an AssertionError if this isn't the case.
        """
        assert XY.shape[1] == 2, "TensorGrid.from_tensor: input must be mn × 2"
        mn = XY.shape[0]
        X = XY[:, 0]
        Y = XY[:, 1]

        x_varies_first = (mn > 1 and X[1] != X[0]).item()
        if x_varies_first:
            X, Y = Y, X

        x0 = X[0]
        n = 0
        while n < mn and X[n] == x0:
            n += 1

        assert (
            mn % n == 0
        ), f"TensorGrid.from_tensor: found len(xs) = {n} which isn't a divisor of {mn}"
        # Then, the first `n` y-values form the basis in the y-direction
        # And the x-basis is given by a strided slice of the x-values
        ys = Y[:n]
        xs = X[::n]

        if x_varies_first:
            xs, ys = ys, xs

        if len(xs) == len(ys) and all(ys == xs):
            # Clear the ys array, so that the TensorGrid constructor instead re-uses the same array
            ys = None

        grid = TensorGrid(xs, ys, x_varies_first)
        assert all(
            grid.full().ravel() == XY.ravel()
        ), "TensorGrid.from_tensor: input was not of tensor product form"
        return grid

    @property
    def grid_shape(self):
        return (self.xs.numel(), self.ys.numel())

    def __str__(self):
        m = self.xs.numel()
        n = self.ys.numel()
        return f"TensorGrid({m}×{n})"

    def __repr__(self):
        return f"TensorGrid({self.xs}, {self.ys})"

    def to(self, device):
        xs = self.xs.to(device)
        ys = self.ys.to(device)
        return TensorGrid(xs, ys, self.x_varies_first)


def gausslegendre(n, dtype=torch.float64, device="cpu"):
    """
    Creates an n-point Gauss-Legendre quadrature rule on the interval [0, 1].
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    nodes = (nodes + 1) / 2
    weights = weights / 2
    nodes = torch.tensor(nodes, dtype=dtype, device=device)
    weights = torch.tensor(weights, dtype=dtype, device=device)
    return nodes, weights


def trapezoid(n, dtype=torch.float64, device="cpu"):
    """
    Creates an n-point trapezoid quadrature rule on the interval [0, 1].
    """
    nodes = torch.linspace(0, 1, n, dtype=dtype, device=device)
    weights = torch.full_like(nodes, nodes[1] - nodes[0])
    weights[0] /= 2
    weights[-1] /= 2
    return nodes, weights


def midpoint(n, dtype=torch.float64, device="cpu"):
    """
    Creates an n-point midpoint quadrature rule on the interval [0, 1].
    """
    nodes = torch.linspace(0, 1, 2 * n + 1, dtype=dtype, device=device)[1::2]
    weights = torch.full_like(nodes, nodes[1] - nodes[0])
    return nodes, weights


def repeated_gauss(n_segments, n_points, dtype=torch.float64, device="cpu"):
    """
    Creates a quadrature rule of n_segments * n_points points on [0, 1], by taking `n_segments`
    copies of the Gauss-Legendre rule with `n_points` points.
    """
    knots = torch.linspace(0, 1, n_segments + 1, dtype=dtype, device=device)
    return repeated_gauss_nonuniform(n_points, knots, dtype, device)


def repeated_gauss_nonuniform(inner, knots, dtype=torch.float64, device="cpu"):
    """
    Creates a quadrature rule of inner * (length(knots) - 1) points on the interval [knots[0],
    knots[-1]], by using a gauss-legendre quadrature rule with `inner` points on each interval
    (knots[i], knots[i + 1]).
    """
    n, w = gausslegendre(inner, dtype=dtype, device=device)
    starts = knots[:-1]
    lens = knots.diff()
    N = n[None, :] * lens[:, None] + starts[:, None]
    W = w[None, :] * lens[:, None]
    return N.flatten(), W.flatten()


def cartesian(nodes, weights):
    """
    Turns an n-point quadrature rule in 1D into an n²-point quadrature rule in 2D, by taking the
    tensor product. The result is a TensorGrid for the nodes, and an n²-vector for the weights.
    """
    nodes_2d = TensorGrid(nodes)
    weight_2d = (weights[:, None] * weights[None, :]).ravel()
    return nodes_2d, weight_2d


def tensorised(
    quad_x: Tuple[torch.Tensor, torch.Tensor],
    quad_y: Union[Tuple[torch.Tensor, torch.Tensor], None] = None,
):
    """
    Given a 1D quadrature rule for the X-dimension, and an optional 1D quadrature rule for the
    Y-dimension (defaults to being the same as the X quadrature), create a 2D quadrature rule. The
    quadrature nodes are a TensorGrid, and the weights are a tensor or array (depending on the types
    of the inputs).
    """
    symmetric = quad_y is None
    nodes_x, weights_x = quad_x
    nodes_y, weights_y = (nodes_x, weights_x) if symmetric else quad_y
    nodes_2d = TensorGrid(nodes_x, None if symmetric else nodes_y)
    # The TensorGrid is constructed with x_varies_first = False
    # So y should vary first when flattening
    # This should also be the case for the weights
    # Since PyTorch tensors and NumPy arrays are stored row-major, this
    # means y should be the row-index, and x should be the column index
    weights_2d = (weights_x[:, None] * weights_y[None, :]).ravel()
    return nodes_2d, weights_2d


def get_quadrature(name, dtype=torch.float64, device="cpu"):
    if match := re.match(r"t(\d+)$", name):
        # Trapezoidal rule
        n = int(match.group(1))
        return trapezoid(n, dtype, device)
    elif match := re.match(r"m(\d+)$", name):
        # Midpoint rule
        n = int(match.group(1))
        return midpoint(n, dtype, device)
    elif match := re.match(r"gl(\d+)$", name):
        # Gauss-Legendre rule
        n = int(match.group(1))
        return gausslegendre(n, dtype, device)
    elif match := re.match(r"gl(\d+)x(\d+)$", name):
        # Repeated Gauss-Legendre rule
        # The first number is the number of segments (outer)
        # The second number is the number of points per segment (inner)
        n = int(match.group(1))
        m = int(match.group(2))
        return repeated_gauss(n, m, dtype, device)
