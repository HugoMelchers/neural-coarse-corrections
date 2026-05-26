# This file contains a number of classes representing function bases for 1D or 2D domains.
# - 1D: B-Spline
# - 2D: Tensor product of 1D bases
# - Functions (linear combinations of basis functions)
#
# They all follow the same interface, and can be called as functions:
# 1D:
# - f.forward(xs) -> F × N
# - f.derivative(xs, k=1) -> F × N
# at least for k = 1 and k = 2
# where xs is a 1D array of length N, and F = len(f) is the number of basis functions
#
# 2D:
# - f.forward(points) -> F × N
# - f.gradient(points) -> F × 2 × N (x-derivative, then y-derivative)
# - f.hessian(points) -> F × 3 × N (2nd derivatives in order xx, xy, yy)
# where points is either a N×2 array or a TensorGrid,
# and F = len(f) is the number of basis functions

import torch
from torch import nn
from math import log, exp, ceil, factorial
from ...utils.quadrature import gausslegendre, TensorGrid, repeated_gauss_nonuniform
from ...utils.misc import einsum

# TODO:
# - Make each basis have a `dtype`


class Functions(nn.Module):
    def __init__(self, basis, weights):
        super().__init__()
        """
        Construct a set of functions using different weights in the same basis. If `basis` consists
        of B functions, then `weights` should be a `F×B` array and the resulting `Functions` object
        will represent `F` functions.
        """
        self.basis = basis
        self.register_buffer("weights", weights)
        self.dim = self.basis.dim

    def is_singleton(self):
        return self.weights.ndim == 1

    def get_weights_2d(self):
        if self.is_singleton():
            # Add a new axis so we have an axis for the output functions
            return self.weights[None]
        else:
            return self.weights

    def maybe_flatten(self, output):
        if self.is_singleton():
            return output.squeeze(0)
        else:
            return output

    def forward(self, points):
        weights = self.get_weights_2d()

        if isinstance(points, TensorGrid) and isinstance(self.basis, TensorBasis):
            # Special path: for tensor basis on tensor grid, we do not need to first evaluate the
            # entire tensor product. Instead, we evaluate the components, and then do tensor product
            # + linear combinations in one step
            f_x = self.basis.f_x
            f_y = self.basis.f_y if self.basis.f_y is not None else f_x
            p_x = points.xs
            p_y = points.ys if points.ys is not None else p_x
            W = weights.reshape(-1, len(f_x), len(f_y))
            fxs = f_x(p_x)
            fys = fxs if f_y is f_x and p_y is p_x else f_y(p_y)
            output = einsum("fx,Ffg,gy->Fxy", fxs, W, fys)
            if points.x_varies_first:
                output = output.permute(0, 2, 1)
            output = output.reshape(output.shape[0], -1)
        else:
            F = self.basis(points)
            output = einsum("bx,fb->fx", F, weights)
        return self.maybe_flatten(output)

    def derivative(self, points, k=1):
        # constant term drops out in the derivatives
        F = self.basis.derivative(points, k)
        return self.maybe_flatten(einsum("bx,fb->fx", F, self.get_weights_2d()))

    def gradient(self, points):
        weights = self.get_weights_2d()
        if isinstance(points, TensorGrid) and isinstance(self.basis, TensorBasis):
            f_x = self.basis.f_x
            f_y = self.basis.f_y if self.basis.f_y is not None else f_x
            p_x = points.xs
            p_y = points.ys if points.ys is not None else p_x
            W = weights.reshape(-1, len(f_x), len(f_y))
            fxs = f_x(p_x)
            fys = fxs if f_y is f_x and p_y is p_x else f_y(p_y)
            d_fxs = f_x.derivative(p_x)
            d_fys = d_fxs if f_y is f_x and p_y is p_x else f_y.derivative(p_y)
            output_x = einsum("fx,Ffg,gy->Fxy", d_fxs, W, fys)
            output_y = einsum("fx,Ffg,gy->Fxy", fxs, W, d_fys)
            if points.x_varies_first:
                output_x = output_x.permute(0, 2, 1)
                output_y = output_y.permute(0, 2, 1)
            output_x = output_x.reshape(output_x.shape[0], -1)
            output_y = output_y.reshape(output_y.shape[0], -1)
            return self.maybe_flatten(torch.stack((output_x, output_y), dim=1))
        else:
            F = self.basis.gradient(points)
            return self.maybe_flatten(einsum("bix,fb->fix", F, weights))

    def hessian(self, points):
        weights = self.get_weights_2d()
        if isinstance(points, TensorGrid) and isinstance(self.basis, TensorBasis):
            f_x = self.basis.f_x
            f_y = self.basis.f_y if self.basis.f_y is not None else f_x
            p_x = points.xs
            p_y = points.ys if points.ys is not None else p_x
            W = weights.reshape(-1, len(f_x), len(f_y))
            fxs = f_x(p_x)
            fys = fxs if f_y is f_x and p_y is p_x else f_y(p_y)
            d_fxs = f_x.derivative(p_x)
            d_fys = d_fxs if f_y is f_x and p_y is p_x else f_y.derivative(p_y)
            d2_fxs = f_x.derivative(p_x, 2)
            d2_fys = d2_fxs if f_y is f_x and p_y is p_x else f_y.derivative(p_y, 2)
            output_xx = einsum("fx,Ffg,gy->Fxy", d2_fxs, W, fys)
            output_xy = einsum("fx,Ffg,gy->Fxy", d_fxs, W, d_fys)
            output_yy = einsum("fx,Ffg,gy->Fxy", fxs, W, d2_fys)
            if points.x_varies_first:
                output_xx = output_xx.permute(0, 2, 1)
                output_xy = output_xy.permute(0, 2, 1)
                output_yy = output_yy.permute(0, 2, 1)
            output_xx = output_xx.reshape(output_xx.shape[0], -1)
            output_xy = output_xy.reshape(output_xy.shape[0], -1)
            output_yy = output_yy.reshape(output_yy.shape[0], -1)
            return self.maybe_flatten(torch.stack((output_xx, output_xy, output_yy), dim=1))
        else:
            F = self.basis.hessian(points)
            return self.maybe_flatten(einsum("bix,fb->fix", F, weights))

    def __getitem__(self, ids):
        return Functions(self.basis, self.weights[ids])

    def __len__(self):
        if self.is_singleton():
            return None
        else:
            return self.weights.shape[0]


class TensorBasis(nn.Module):
    dim = 2

    def __init__(self, f_x, f_y=None):
        """
        Create a 2D basis from a tensor product of 1D function bases.
        - `f_x`: the basis functions to use along the `x`-direction
        - `f_y`: the basis functions to use along the `y`-direction. Set to `None` to use `f_x`
          along both axes

        The resulting 2D function basis `f_xy` contains a tensor product of the basis functions
        `f_x` and `f_y`, ordered y-varies-first, i.e.

            f_xy(x, y)[n_y * i + j] = f_x[i](x) + f_y[j](y)
        """
        super().__init__()
        self.f_x = f_x
        self.f_y = f_y or f_x

    def forward(self, points):
        f_x = self.f_x
        f_y = self.f_y
        if isinstance(points, TensorGrid):
            f = f_x(points.xs)
            g = f_y(points.ys)
            return self.tensor_product(f, g, points.x_varies_first)
        else:
            X = points[:, 0]
            Y = points[:, 1]
            f = f_x(X)
            g = f_y(Y)
            return self.outer_product(f, g)

    def gradient(self, points):
        f_x = self.f_x
        f_y = self.f_y
        if isinstance(points, TensorGrid):
            f = f_x(points.xs)
            df = f_x.derivative(points.xs)
            g = f_y(points.ys)
            dg = f_y.derivative(points.ys)
            return torch.stack(
                (
                    self.tensor_product(df, g, points.x_varies_first),
                    self.tensor_product(f, dg, points.x_varies_first),
                ),
                1,
            )
        else:
            X = points[:, 0]
            Y = points[:, 1]
            f = f_x(X)
            g = f_y(Y)
            df = f_x.derivative(X)
            dg = f_y.derivative(Y)
            return torch.stack((self.outer_product(df, g), self.outer_product(f, dg)), 1)

    def hessian(self, points):
        f_x = self.f_x
        f_y = self.f_y
        if isinstance(points, TensorGrid):
            f = f_x(points.xs)
            df = f_x.derivative(points.xs)
            ddf = f_x.derivative(points.xs, 2)
            g = f_y(points.ys)
            dg = f_y.derivative(points.ys)
            ddg = f_y.derivative(points.ys, 2)
            return torch.stack(
                (
                    self.tensor_product(ddf, g, points.x_varies_first),
                    self.tensor_product(df, dg, points.x_varies_first),
                    self.tensor_product(f, ddg, points.x_varies_first),
                ),
                1,
            )
        else:
            X = points[:, 0]
            Y = points[:, 1]
            f = f_x(X)
            g = f_y(Y)
            df = f_x.derivative(X)
            dg = f_y.derivative(Y)
            ddf = f_x.derivative(X, 2)
            ddg = f_y.derivative(Y, 2)
            return torch.stack(
                (
                    self.outer_product(ddf, g),
                    self.outer_product(df, dg),
                    self.outer_product(f, ddg),
                ),
                1,
            )

    def gram_matrix(self):
        Mx = self.f_x.gram_matrix()
        My = self.f_y.gram_matrix()
        return torch.kron(Mx, My)

    def inverse_gram_matrix(self):
        Mxi = self.f_x.inverse_gram_matrix()
        Myi = self.f_y.inverse_gram_matrix()
        return torch.kron(Mxi, Myi)

    @staticmethod
    def outer_product(a, b):
        # a should be f × n
        # b should be g × n
        # output should then be (f*g) × n
        f = a.shape[0]
        g = b.shape[0]
        n = a.shape[1]
        return (a.reshape(f, 1, n) * b.reshape(1, g, n)).reshape(f * g, n)

    @staticmethod
    def tensor_product(a, b, x_varies_first):
        # a should be f × m
        # b should be g × n
        # output should then be fg × mn
        f, m = a.shape
        g, n = b.shape
        if x_varies_first:
            # when reshaping: last index varies first. We therefore now want the last indices to be
            # m, and g
            a = a.reshape(f, 1, 1, m)
            b = b.reshape(1, g, n, 1)
        else:
            a = a.reshape(f, 1, m, 1)
            b = b.reshape(1, g, 1, n)

        return (a * b).reshape(f * g, m * n)

    def __len__(self):
        f_x = self.f_x
        f_y = self.f_y or f_x
        return len(f_x) * len(f_y)


class BSpline(nn.Module):
    dim = 1

    def __init__(self, lims, n_segments, degree, max_ratio=0.2, log_widths=None):
        """
        Creates a new BSpline that forms a basis of the interval [a, b], split up into `nr_segments`
        intervals. The B-splines have the given `degree`. Note that the total number of functions is
        then not equal to `n_segments`, but `n_segments + degree`.

        The argument `max_ratio` affects the random initialisation of the B-splines. This gives
        the maximum allowed ratio of the largest to shortest segments that partition the interval
        [a, b]. If max_ratio=1, then all intervals must have the same length, meaning that the
        initialisation is uniform. By default, the largest segment is at most 5 times as long as
        the shortest interval, which corresponds to max_ratio=0.2 but can also be achieved with
        max_ratio=5.

        If `log_widths` is given, then the returned layer is not trainable, but instead contains the
        specified log_widths. In this case, the length of this array should equal `n_segments`, and
        the value of max_ratio, if present, will be ignored.

        For example, BSpline((0, 1), 5, 3, 1) creates a B-spline basis with 5 equal knot spans and
        cubic splines, so 8 basis functions in total, all supported on the interval [0, 1].
        """
        super(BSpline, self).__init__()
        self.n_segments = n_segments
        self.degree = degree
        self.lims = lims
        if log_widths is not None:
            self.dtype = log_widths.dtype
            assert log_widths.numel() == n_segments, (
                f"BSplineLayer got n_segments={n_segments}, but log_widths "
                f"with {log_widths.numel()} elements"
            )
            if isinstance(log_widths, nn.Parameter):
                self.log_widths = log_widths
            elif isinstance(log_widths, torch.Tensor):
                self.register_buffer("log_widths", log_widths)
                knot_vector = self.get_knot_vector()
                self.register_buffer("knot_vector", knot_vector)
            else:
                raise TypeError(
                    "BSplineLayer1D: expected `log_widths` to be None, a nn.Parameter, "
                    f"or torch.Tensor (got type {type(log_widths)})"
                )
        else:
            self.log_widths = nn.Parameter(torch.rand(n_segments) * log(max_ratio))
            self.dtype = torch.float32

    @staticmethod
    def uniform(lims, n_segments, degree, dtype=torch.float32):
        return BSpline(lims, n_segments, degree, log_widths=torch.zeros((n_segments,), dtype=dtype))

    @staticmethod
    def from_parts(degree, knots):
        if not isinstance(knots, torch.Tensor):
            knots = torch.tensor(knots)

        # Assume open knot vector, so we remove the first and last `degree` duplicate knots
        log_widths = knots[degree:-degree].diff().log()
        lims = (knots[0], knots[-1])
        n_segments = log_widths.numel()
        return BSpline(lims, n_segments, degree, log_widths=log_widths)

    def quadrature(self, deg):
        """
        Get a quadrature rule that is exact for functions that are piecewise polynomials that are
        defined on the same segments as this BSplineLayer and piecewise of degree at most `deg`.
        """
        nr_nodes = ceil((deg + 1) / 2)
        segs = self.get_partition()
        return repeated_gauss_nonuniform(nr_nodes, segs, self.dtype, self.log_widths.device)

    def get_knot_vector(self):
        """
        Gets the knot vector of the B-spline layer. The knot vector is not directly a parameter
        of the model, because the distance between subsequent knots should not become too small.
        Furthermore, the first and last `degree + 1` should always equal the left and right endpoint
        of the interval [a, b], respectively. For these reasons, the model learns the logarithms of
        the widths of the intervals, up to some constant.
        """
        if hasattr(self, "knot_vector"):
            return self.knot_vector
        widths = self.log_widths.exp()
        a, b = self.lims
        widths = widths * (b - a) / widths.sum()
        knots_internal = widths.cumsum_(0) + a
        device = knots_internal.device
        knots = torch.cat(
            (
                torch.full((self.degree + 1,), a, device=device, dtype=self.dtype),
                knots_internal,
                torch.full((self.degree,), b, device=device, dtype=self.dtype),
            )
        )
        return knots

    def get_partition(self):
        """
        Gets the points that partition the interval on which this function is defined. This is
        the same as the knot vector, except that the multiplicities of the end points are just 1,
        instead of being higher depending on the degree.
        """
        widths = self.log_widths.exp()
        a, b = self.lims
        widths = widths * (b - a) / widths.sum()
        knots_internal = widths.cumsum_(0) + a
        knots = torch.cat(
            (
                torch.full((1,), a, device=knots_internal.device, dtype=self.dtype),
                knots_internal,
            )
        )
        return knots

    def forward(self, x):
        """
        Compute the b-splines at the given x-values. The output is an N×M array, where N = length(x)
        and M is the number of basis functions.
        """
        knots = self.get_knot_vector()
        out = BSpline._bspline_get_values(knots, self.degree, x)
        return out

    def derivative(self, x, k=1):
        """
        Compute the k-th derivative (default 1) of the basis functions on the given points.
        """
        degree = self.degree
        knots = self.get_knot_vector()
        nf = knots.numel() - degree - 1
        W = torch.diag(torch.ones(nf, dtype=x.dtype, device=x.device))
        out = BSpline._bspline_get_derivatives(knots, degree, k, x, W)
        return out

    def derivative_func(self, k=1):
        """
        Compute the k-th derivative (default 1) of the basis functions, and express the result as a
        new BSplineLayer that can be called as a function.
        """
        new_degree, new_weights = BSpline._bspline_get_derivative_weights(
            self.get_knot_vector(), self.degree, k, None
        )
        basis = BSpline(
            self.lims,
            self.n_segments,
            new_degree,
            max_ratio=1.0,
            log_widths=self.log_widths,
        )
        return Functions(basis, new_weights)

    def derivatives(self, x, weights):
        """
        Computes the linear combination of derivatives:

            weights[0] * f(x) + weights[1] * f'(x) + weights[2] * f''(x) + ...

        for each of the basis functions.
        """
        result = weights[0] * self(x)
        for k in range(1, len(weights)):
            result = result + weights[k] * self.derivative(x, k)

        return result.T

    def gram_matrix(self):
        """
        Computes the Gram matrix of the basis, i.e. the matrix M where

            Mᵢⱼ = ∫ fᵢ(x)fⱼ(x) dx
        """
        n, w = self.quadrature(2 * self.degree)
        u = self(n)
        return torch.einsum("fx,gx,x->fg", u, u, w)

    def inverse_gram_matrix(self):
        return torch.inverse(self.gram_matrix())

    def integrate_deriv_square(self, with_cross_terms=False):
        """
        Compute the integrals of f'(x)² dx over the domain of the basis. If with_cross_terms=True,
        also compute f_i'(x) f_j'(x) dx, and return an n*n matrix instead of a vector.
        """
        n, w = self.quadrature(2 * self.degree - 2)
        u = self.derivative(n)
        M = w * u @ u.T
        if with_cross_terms:
            return M
        else:
            return M.diag()

    @staticmethod
    def _bspline_get_values(knots, degree, x):
        Nx = x.numel()
        x = x.unsqueeze(0)
        padding = torch.full((degree, Nx), False, device=x.device)
        knots = knots.unsqueeze(1)
        result = torch.cat(
            (
                padding,
                x < knots[degree + 1, :],
                (knots[degree + 1 : -degree - 2, :] <= x)
                * (x < knots[degree + 2 : -degree - 1, :]),
                knots[-degree - 2, :] <= x,
                padding,
            ),
            dim=0,
        ).to(x.dtype)
        eps = torch.tensor(1e-10, dtype=x.dtype, device=x.device)
        for deg in range(1, degree + 1):
            nf = knots.numel() - deg - 1
            a = knots[0 : nf + 1]
            b = knots[deg : (1 + deg + nf)]
            l = torch.maximum(b - a, eps)
            v1 = result[0:nf] * (x - a[:-1]) / l[:-1]
            v2 = result[1 : (nf + 1)] * (b[1:] - x) / l[1:]
            result = v1 + v2

        return result

    @staticmethod
    def _bspline_get_derivatives(knots, degree, k, x, W):
        nf = knots.numel() - degree - 1
        if k == 0:
            return W @ BSpline._bspline_get_values(knots, degree, x)
        elif k > degree:
            return torch.zeros((nf, x.numel()), device=x.device)
        else:
            # The derivatives of a degree-p B-spline basis can be expressed as linear combinations
            # of degree (p-1) B-spline basis functions. To evaluate the lower-order functions,
            # I also remove the first and last knot, since the lower degree also expects lower
            # multiplicities of the start and end knot.
            eps = torch.tensor(1e-10, dtype=x.dtype, device=x.device)
            a1 = knots[1:nf]
            b1 = knots[degree + 1 : -1]
            W2 = degree * W.diff(dim=1) / torch.maximum(b1 - a1, eps).unsqueeze(0)
            return BSpline._bspline_get_derivatives(knots[1:-1], degree - 1, k - 1, x, W2)

    @staticmethod
    def _bspline_get_derivative_weights(knots, degree, k, W):
        nf = knots.numel() - degree - 1
        if W is None:
            W = torch.diag(torch.ones(nf, device=knots.device, dtype=knots.dtype))
        if k == 0:
            return (degree, W)
        elif k > degree:
            return (degree, torch.zeros_like(W))
        else:
            eps = torch.tensor(1e-10, dtype=knots.dtype, device=knots.device)
            a1 = knots[1:nf]
            b1 = knots[degree + 1 : -1]
            W2 = degree * W.diff(dim=1) / torch.maximum(b1 - a1, eps).unsqueeze(0)
            # When we write the derivative in terms of lower-degree B-splines, the current
            # formulation will contain some basis functions that have zero support since their
            # control points are just the left or right endpoint repeated. So we have to remove
            # these.
            return BSpline._bspline_get_derivative_weights(knots[1:-1], degree - 1, k - 1, W2)

    def __len__(self):
        return self.degree + self.n_segments


class RBF1D(nn.Module):
    # Implementation of Radial Basis Functions (RBFs) in pytorch. Based on
    # <https://github.com/rssalessio/PytorchRBFLayer>.
    dim = 1

    def __init__(self, num, centres=None, log_sigmas=None):
        super(RBF1D, self).__init__()
        self.num = num
        if centres is not None:
            assert log_sigmas is not None, "RBF1D: centres is not None, but log_sigmas is None"
            self.register_buffer("centres", centres)
            self.register_buffer("log_sigmas", log_sigmas)
        else:
            assert log_sigmas is None, "RBF1D: centres is None, but log_sigmas is not None"
            self.centres = nn.Parameter(torch.Tensor(num))
            self.log_sigmas = nn.Parameter(torch.Tensor(num))
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, log(0.5))

    def forward(self, input):
        d = input[None, :] - self.centres[:, None]
        r = d / self.log_sigmas.exp()[:, None]
        return (-r.square()).exp()

    def derivative(self, input, k=1):
        # The derivatives of the 1D Gaussian RBF can be computed as
        # dᵏ/dxᵏ exp(-(x - μ)²/σ²) = (-1)ᵏ/σᵏ exp(-(x - μ)²/σ²) Hₖ((x - μ) / σ)
        d = input[None, :] - self.centres[:, None]
        sigma = self.log_sigmas.exp()[:, None]
        sigma_sq = sigma.square()
        x = d / sigma
        f = (-x.square()).exp()
        if k == 1:
            H1 = 2 * x
            return -f * H1 / sigma
        elif k == 2:
            H2 = 4 * x**2 - 2
            return f * H2 / sigma_sq

    def __str__(self):
        return f"RBF1D({self.num})"

    def __len__(self):
        return self.log_sigmas.shape[0]


class ZeroBasis(nn.Module):
    def __init__(self, count, dim):
        super().__init__()
        self.count = count
        self.dim = dim

    def forward(self, points):
        p = points.shape[0]
        return torch.zeros((self.count, p), dtype=points.dtype, device=points.device)

    def derivative(self, points, k=1):
        return self(points)

    def __len__(self):
        return self.count


class OneBasis(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 1

    def forward(self, points):
        p = points.shape[0]
        return torch.ones((1, p), dtype=points.dtype, device=points.device)

    def derivative(self, points, k=1):
        p = points.shape[0]
        return torch.zeros((1, p), dtype=points.dtype, device=points.device)

    def __len__(self):
        return 1
