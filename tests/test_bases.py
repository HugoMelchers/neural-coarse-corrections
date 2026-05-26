import unittest
from src.training.subnets.basis import BSpline, RBF1D, TensorBasis, Functions, ZeroBasis, OneBasis
import torch
from src.utils.misc import rms, relerr
from src.utils.quadrature import TensorGrid


def err(a, b):
    if a.shape != b.shape:
        raise RuntimeError(
            f"err(a, b) expects tensors of the same shape, got {a.shape=}, {b.shape=}"
        )
    return rms(a - b)


def is_close(a, b, tol):
    return err(a, b) <= tol * max(rms(a), rms(b))


def make_bases_1d():
    torch.manual_seed(1234)
    return [
        ("Zeros", ZeroBasis(7, 1)),
        ("Ones", OneBasis()),
        ("BSpline", BSpline((0, 1), 3, 3)),
        ("RBF1D", RBF1D(8)),
        (
            "BSpline functions",
            Functions(BSpline((0, 1), 3, 3), torch.randn(13, 6)),
        ),
    ]


def make_bases_2d():
    torch.manual_seed(1234)
    return [
        (
            "Tensor BSpline x RBF1D",
            TensorBasis(BSpline.uniform((0, 1), 3, 3), RBF1D(6)),
        ),
        (
            "Functions BSpline x RBF1D",
            Functions(
                TensorBasis(BSpline.uniform((0, 1), 3, 3), RBF1D(6)),
                torch.randn(7, 36),
            ),
        ),
    ]


def first_derivative_error(basis, n):
    with torch.no_grad():
        xs = torch.linspace(0, 1, n)
        h = 1 / (n - 1)
        f = basis(xs)
        df = basis.derivative(xs[1:-1])
        df_num = (f[:, 2:] - f[:, :-2]) / (2 * h)
        return err(df, df_num)


def second_derivative_error(basis, n):
    with torch.no_grad():
        xs = torch.linspace(0, 1, n)
        h = 1 / (n - 1)
        f = basis(xs)
        d2f = basis.derivative(xs[1:-1], 2)
        d2f_num = (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / h**2
        return err(d2f, d2f_num)


class BasesTest(unittest.TestCase):
    def test_count(self):
        n = 10
        xs = torch.linspace(0, 1, n)
        for name, basis in make_bases_1d():
            with self.subTest(name):
                with torch.no_grad():
                    f = basis(xs)
                    self.assertEqual(f.shape[0], len(basis))
                    self.assertEqual(f.shape[1], n)

    def test_1d_derivatives(self):
        n = 200
        xs = torch.linspace(0, 1, n)
        h = 1 / (n - 1)
        for name, basis in make_bases_1d():
            with self.subTest(name):
                with torch.no_grad():
                    f = basis(xs)
                    df = basis.derivative(xs[1:-1], 1)
                    df_num = (f[:, 2:] - f[:, :-2]) / (2 * h)
                    d2f = basis.derivative(xs[1:-1], 2)
                    d2f_num = (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / h**2
                    self.assertTrue(is_close(df, df_num, 1e-3))
                    self.assertTrue(is_close(d2f, d2f_num, 1e-2))

    def test_2d_derivatives(self):
        n = 100
        xs = torch.linspace(0, 1, n)
        h = 1 / (n - 1)
        g = TensorGrid(xs, None)
        for name, basis in make_bases_2d():
            with self.subTest(name):
                with torch.no_grad():
                    f = basis(g).reshape(-1, n, n)
                    gf = basis.gradient(g).reshape(-1, 2, n, n)
                    Hf = basis.hessian(g).reshape(-1, 3, n, n)
                    gf_x_num = (f[:, 2:, :] - f[:, :-2, :]) / (2 * h)
                    gf_y_num = (f[:, :, 2:] - f[:, :, :-2]) / (2 * h)
                    Hf_xx_num = (f[:, 2:, :] - 2 * f[:, 1:-1, :] + f[:, :-2, :]) / (h**2)
                    Hf_xy_num = (f[:, 2:, 2:] - f[:, 2:, :-2] - f[:, :-2, 2:] + f[:, :-2, :-2]) / (
                        4 * h**2
                    )
                    Hf_yy_num = (f[:, :, 2:] - 2 * f[:, :, 1:-1] + f[:, :, :-2]) / (h**2)
                    self.assertTrue(is_close(gf[:, 0, 1:-1, :], gf_x_num, 2e-3))
                    self.assertTrue(is_close(gf[:, 1, :, 1:-1], gf_y_num, 1e-3))

                    self.assertTrue(is_close(Hf[:, 0, 1:-1, :], Hf_xx_num, 1e-2))
                    self.assertTrue(is_close(Hf[:, 1, 1:-1, 1:-1], Hf_xy_num, 1e-2))
                    self.assertTrue(is_close(Hf[:, 2, :, 1:-1], Hf_yy_num, 1e-2))

    def test_2d_tensor_values(self):
        with torch.no_grad():
            xs = torch.linspace(0, 1, 11)
            ys = torch.linspace(0, 1, 6)
            grids = [
                TensorGrid(xs),
                TensorGrid(xs, ys),
                TensorGrid(xs, x_varies_first=True),
                TensorGrid(xs, ys, x_varies_first=True),
            ]
            grids_full = [grid.full() for grid in grids]
            grid_names = ["x", "xy", "x 2", "xy 2"]
            for name, basis in make_bases_2d():
                with self.subTest(name):
                    for i in range(4):
                        with self.subTest(grid_names[i]):
                            grid = grids[i]
                            grid_full = grids_full[i]
                            y1 = basis(grid)
                            y2 = basis(grid_full)
                            self.assertLess(err(y1, y2), 1e-4 * max(rms(y1), rms(y2)))

                            dy1 = basis.gradient(grid)
                            dy2 = basis.gradient(grid_full)
                            self.assertLess(err(dy1, dy2), 1e-4 * max(rms(dy1), rms(dy2)))

                            ddy1 = basis.hessian(grid)
                            ddy2 = basis.hessian(grid_full)
                            self.assertLess(err(ddy1, ddy2), 1e-4 * max(rms(ddy1), rms(ddy2)))

    def test_function_indexing_1d(self):
        torch.manual_seed(1234)
        f = Functions(BSpline((0, 1), 3, 3), torch.randn(13, 6))
        xs = torch.linspace(0, 1, 21)
        self.assertTrue(is_close(f[:2](xs), f(xs)[:2], 1e-5))
        self.assertTrue(is_close(f[:2].derivative(xs), f.derivative(xs)[:2], 1e-5))
        self.assertTrue(is_close(f[:2].derivative(xs, 2), f.derivative(xs, 2)[:2], 1e-5))

    def test_function_indexing_2d(self):
        torch.manual_seed(1234)
        f = make_bases_2d()[1][1]
        xs = torch.rand((100, 2))
        self.assertTrue(is_close(f[:2](xs), f(xs)[:2], 1e-5))
        self.assertTrue(is_close(f[:2].gradient(xs), f.gradient(xs)[:2], 1e-5))
        self.assertTrue(is_close(f[:2].hessian(xs), f.hessian(xs)[:2], 1e-5))

    def test_function_indexing_1d_single(self):
        torch.manual_seed(1234)
        f = Functions(BSpline((0, 1), 3, 3), torch.randn(13, 6))
        xs = torch.linspace(0, 1, 21)
        self.assertTrue(is_close(f[2](xs), f(xs)[2], 1e-5))
        self.assertTrue(is_close(f[2].derivative(xs), f.derivative(xs)[2], 1e-5))
        self.assertTrue(is_close(f[2].derivative(xs, 2), f.derivative(xs, 2)[2], 1e-5))

    def test_function_indexing_2d_single(self):
        torch.manual_seed(1234)
        f = make_bases_2d()[1][1]
        xs = torch.rand((100, 2))
        self.assertTrue(is_close(f[0](xs), f(xs)[0], 1e-5))
        self.assertTrue(is_close(f[0].gradient(xs), f.gradient(xs)[0], 1e-5))
        self.assertTrue(is_close(f[0].hessian(xs), f.hessian(xs)[0], 1e-5))


class BSplineTest(unittest.TestCase):
    def test_weights_derivative(self):
        torch.manual_seed(1234)
        basis = BSpline((0, 1), 10, 2)
        xs = torch.linspace(0, 1, 201)
        dy_1 = basis.derivative(xs)
        dy_2 = basis.derivative_func()(xs)
        self.assertTrue(is_close(dy_2, dy_1, 1e-7))

    def test_weights_derivative_2(self):
        torch.manual_seed(1234)
        basis = BSpline((0, 1), 10, 2)
        xs = torch.linspace(0, 1, 201)
        dy_1 = basis.derivative(xs, 2)
        dy_2 = basis.derivative_func(2)(xs)
        self.assertTrue(is_close(dy_2, dy_1, 1e-7))
