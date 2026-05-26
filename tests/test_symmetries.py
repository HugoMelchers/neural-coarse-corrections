# A test file to check that symmetry group averaging (abbreviated as SGA) works. This means that a
# model with SGA enabled should be equivariant to mirroring the inputs horizontally or vertically,
# i.e. mirroring the input horizontally also produces an output that is exactly the same but
# mirrored horizontally, etc.
import unittest

import torch
from torch import nn

from src.training.models import NGO, DeepONet, VarMiON, RINO
from src.utils.quadrature import get_quadrature
from src.training.subnets.misc import ReshapeLayer, BoundedActivation, StrictReshapeLayer
from src.training.subnets.unet import UNet
from src.training.subnets.basis import BSpline, TensorBasis
from src.utils.misc import relerr, rms


def make_ngo(problem):
    if problem == "diffusion" or problem == "helmholtz":
        n_in = 1
    elif problem == "advection-diffusion":
        n_in = 3
    basis = TensorBasis(BSpline.uniform((0, 1), 9, 3))
    C = 18
    system_net = nn.Sequential(
        UNet.make(
            in_channels=n_in,
            out_channels=24,
            channels=[24, 24, 24, 24],
            lifting_dim=24,
            projection_dim=None,
            activation="leakyrelu",
            widths=[2, 2, 3],
            strides=[2, 2, 3],
            pointwise_dim=None,
            input_size=[12, 12],
            skip_type="add",
            bias=True,
        ),
        nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=3, stride=3),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=2, stride=2),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels=24, out_channels=1, kernel_size=2, stride=2),
        StrictReshapeLayer((1, 144, 144), (144, 144)),
    )
    model = NGO(problem=problem, system_net=system_net, basis=basis)
    return model


def make_deeponet(problem):
    # 12×12 grid for f and theta, 2× 2×12 for g and eta
    n_inputs = 2 * 12**2 + 2 * 2 * 12
    if problem == "advection-diffusion":
        # Additionally 2× 12×12 for v_x and v_y
        n_inputs += 2

    basis = TensorBasis(BSpline.uniform((0, 1), 9, 3))
    branch = nn.Sequential(
        nn.Linear(n_inputs, 60),
        nn.LeakyReLU(),
        nn.Linear(60, 12**2),
    )
    return DeepONet(problem, branch, basis, config={})


def make_varmion(problem):
    basis = TensorBasis(BSpline.uniform((0, 1), 9, 3))
    linears = nn.Linear(192, 144, bias=False)
    in_channels = 3 if problem == "advection-diffusion" else 1
    branch = nn.Sequential(
        StrictReshapeLayer((144 * in_channels,), (in_channels, 12, 12)),
        UNet.make(
            in_channels=in_channels,
            out_channels=12,
            channels=[12, 12, 12, 12],
            lifting_dim=12,
            projection_dim=None,
            activation="leakyrelu",
            widths=[2, 2, 3],
            strides=[2, 2, 3],
            pointwise_dim=None,
            input_size=[12, 12],
            skip_type="add",
            bias=True,
        ),
        nn.ConvTranspose2d(in_channels=12, out_channels=12, kernel_size=3, stride=3),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels=12, out_channels=12, kernel_size=2, stride=2),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(in_channels=12, out_channels=1, kernel_size=2, stride=2),
        StrictReshapeLayer((1, 144, 144), (144, 144)),
    )
    config = {}
    outputs = "values"
    model = VarMiON(problem, linears, branch, basis, config, outputs)
    return model


def make_rino(problem):
    # 12×12 grid for f and theta, 2× 2×12 for g and eta
    n_inputs = 2 * 12**2 + 2 * 2 * 12
    if problem == "advection-diffusion":
        # Additionally 2× 12×12 for v_x and v_y
        n_inputs += 2

    basis = TensorBasis(BSpline.uniform((0, 1), 9, 3))
    branch = nn.Sequential(
        nn.Linear(n_inputs, 60),
        nn.LeakyReLU(),
        nn.Linear(60, 12**2),
    )
    return RINO(problem, branch, basis, config={})


def make_models(problem, sga, scale):
    models = [
        ("NGO", make_ngo(problem)),
        ("DeepONet", make_deeponet(problem)),
        ("VarMiON", make_varmion(problem)),
        ("RINO", make_rino(problem)),
    ]
    for _, model in models:
        model.config["sga"] = sga
        model.config["scale"] = scale

    return models


def make_random_batch(problem, B, quad_x, quad_y):
    nodes_x, weights_x = quad_x
    nodes_y, weights_y = quad_y
    nx, ny = len(nodes_x), len(nodes_y)
    batch = {}
    names_2d = ["theta", "f", "u"]
    if problem == "advection-diffusion":
        names_2d += ["v_x", "v_y"]

    names_1d_x = ["eta"] + names_2d
    names_1d_y = ["g"] + names_2d
    for name in names_2d:
        batch[name] = torch.randn(B, nx, ny)
        batch[name + "/grad"] = torch.randn(B, 2, nx, ny)
        batch[name + "/nodes_x"] = nodes_x
        batch[name + "/nodes_y"] = nodes_y
        batch[name + "/weights_x"] = weights_x
        batch[name + "/weights_y"] = weights_y

    # Constant velocity field
    if problem == "advection-diffusion":
        batch["v_x"].fill_(batch["v_x"][0, 0, 0])
        batch["v_y"].fill_(batch["v_y"][0, 0, 0])
        batch["v_x/grad"].fill_(0)
        batch["v_y/grad"].fill_(0)

    for name in names_1d_x:
        batch["bottom:" + name] = torch.randn(B, nx)
        batch["top:" + name] = torch.randn(B, nx)
        batch["neumann:" + name] = torch.randn(B, nx, 2)
        batch["bottom:" + name + "/nodes"] = nodes_x
        batch["top:" + name + "/nodes"] = nodes_x
        batch["neumann:" + name + "/nodes"] = nodes_x
        batch["bottom:" + name + "/weights"] = weights_x
        batch["top:" + name + "/weights"] = weights_x
        batch["neumann:" + name + "/weights"] = weights_x

    for name in names_1d_y:
        batch["left:" + name] = torch.randn(B, ny)
        batch["right:" + name] = torch.randn(B, ny)
        batch["dirichlet:" + name] = torch.randn(B, 2, ny)
        batch["left:" + name + "/nodes"] = nodes_y
        batch["right:" + name + "/nodes"] = nodes_y
        batch["dirichlet:" + name + "/nodes"] = nodes_y
        batch["left:" + name + "/weights"] = weights_y
        batch["right:" + name + "/weights"] = weights_y
        batch["dirichlet:" + name + "/weights"] = weights_y

    # Normalise theta to be bounded away from zero
    theta_lo = batch["theta"].min()
    theta_hi = batch["theta"].max()
    batch["theta"] = 0.1 + 0.9 * (batch["theta"] - theta_lo) / (theta_hi - theta_lo)

    batch["nodes_x"] = nodes_x
    batch["nodes_y"] = nodes_y
    batch["weights_x"] = weights_x
    batch["weights_y"] = weights_y
    batch = {k: v.to(torch.float32) for (k, v) in batch.items()}
    batch["problem"] = problem
    return batch


def mirror_x(batch):
    batch = dict(batch)
    problem = batch["problem"]
    names_2d = ["theta", "f", "u"]
    if problem == "advection-diffusion":
        names_2d += ["v_x", "v_y"]
        batch["v_x"] = -batch["v_x"]
        for prefix in ["bottom:", "top:", "left:", "right:", "dirichlet:", "neumann:"]:
            batch[prefix + "v_x"] = -batch[prefix + "v_x"]

    names_1d_x = ["eta"] + names_2d
    names_1d_y = ["g"] + names_2d
    for name in names_2d:
        batch[name] = batch[name].flip(1)

    for name in names_1d_x:
        batch["bottom:" + name] = batch["bottom:" + name].flip(1)
        batch["top:" + name] = batch["top:" + name].flip(1)
        batch["neumann:" + name] = batch["neumann:" + name].flip(1)

    for name in names_1d_y:
        left = batch["left:" + name]
        right = batch["right:" + name]
        batch["left:" + name] = right
        batch["right:" + name] = left
        batch["dirichlet:" + name] = batch["dirichlet:" + name].flip(1)

    return batch


def mirror_y(batch):
    batch = dict(batch)
    problem = batch["problem"]
    names_2d = ["theta", "f", "u"]
    if problem == "advection-diffusion":
        names_2d += ["v_x", "v_y"]
        batch["v_y"] = -batch["v_y"]
        for prefix in ["bottom:", "top:", "left:", "right:", "dirichlet:", "neumann:"]:
            batch[prefix + "v_y"] = -batch[prefix + "v_y"]

    names_1d_x = ["eta"] + names_2d
    names_1d_y = ["g"] + names_2d
    for name in names_2d:
        batch[name] = batch[name].flip(2)

    for name in names_1d_x:
        bottom = batch["bottom:" + name]
        top = batch["top:" + name]
        batch["bottom:" + name] = top
        batch["top:" + name] = bottom
        batch["neumann:" + name] = batch["neumann:" + name].flip(2)

    for name in names_1d_y:
        batch["left:" + name] = batch["left:" + name].flip(1)
        batch["right:" + name] = batch["right:" + name].flip(1)
        batch["dirichlet:" + name] = batch["dirichlet:" + name].flip(2)

    return batch


def scale_params(batch, r):
    # Scale all the parameters of the PDE, so that the solution scales by the reciprocal
    batch = dict(batch)
    problem = batch["problem"]
    names_2d = ["theta"]
    if problem == "advection-diffusion":
        names_2d += ["v_x", "v_y"]

    for name in names_2d:
        batch[name] = batch[name] * r
        batch["bottom:" + name] = batch["bottom:" + name] * r
        batch["top:" + name] = batch["top:" + name] * r
        batch["neumann:" + name] = batch["neumann:" + name] * r
        batch["left:" + name] = batch["left:" + name] * r
        batch["right:" + name] = batch["right:" + name] * r
        batch["dirichlet:" + name] = batch["dirichlet:" + name] * r

    batch["u"] = batch["u"] / r
    batch["bottom:u"] = batch["bottom:u"] / r
    batch["top:u"] = batch["top:u"] / r
    batch["neumann:u"] = batch["neumann:u"] / r
    batch["left:u"] = batch["left:u"] / r
    batch["right:u"] = batch["right:u"] / r
    batch["dirichlet:u"] = batch["dirichlet:u"] / r

    return batch


def scale_rhs(batch, r):
    # Scale all the right-hand side terms of the PDE, so that the solution scales by the same factor
    batch = dict(batch)
    problem = batch["problem"]
    names_2d = ["f", "u"]  # Note that "u" is now not modified
    names_1d_x = ["eta"] + names_2d
    names_1d_y = ["g"] + names_2d
    for name in names_2d:
        batch[name] = batch[name] * r

    for name in names_1d_x:
        batch["bottom:" + name] = batch["bottom:" + name] * r
        batch["top:" + name] = batch["top:" + name] * r
        batch["neumann:" + name] = batch["neumann:" + name] * r

    for name in names_1d_y:
        batch["left:" + name] = batch["left:" + name] * r
        batch["right:" + name] = batch["right:" + name] * r
        batch["dirichlet:" + name] = batch["dirichlet:" + name] * r

    return batch


class SampleIndependenceTest(unittest.TestCase):
    def test_sample_independence_diffusion(self):
        torch.manual_seed(1234)
        models = (
            make_models("diffusion", None, False)
            + make_models("diffusion", None, True)
            + make_models("diffusion", "bspline", False)
            + make_models("diffusion", "bspline", True)
        )
        quad_x = get_quadrature("t12")
        quad_y = get_quadrature("t12")
        batch1 = make_random_batch("diffusion", 2, quad_x, quad_y)
        batch2 = make_random_batch("diffusion", 3, quad_x, quad_y)
        batch12 = {
            k: (
                batch1[k]
                if "nodes" in k or "weights" in k or k == "problem"
                else torch.cat((batch1[k], batch2[k]), dim=0)
            )
            for k in batch1.keys()
        }
        for name, model in models:
            with self.subTest(name):
                with torch.no_grad():
                    out1 = model(batch1)
                    out2 = model(batch2)
                    out12 = model(batch12)

                self.assertLess(relerr(out12[:2], out1), 1e-4)
                self.assertLess(relerr(out12[2:], out2), 1e-4)

    def test_sample_independence_advdiff(self):
        torch.manual_seed(1234)
        models = (
            make_models("advection-diffusion", None, False)
            + make_models("advection-diffusion", None, True)
            + make_models("advection-diffusion", "bspline", False)
            + make_models("advection-diffusion", "bspline", True)
        )
        quad_x = get_quadrature("t12")
        quad_y = get_quadrature("t12")
        batch1 = make_random_batch("advection-diffusion", 2, quad_x, quad_y)
        batch2 = make_random_batch("advection-diffusion", 3, quad_x, quad_y)
        batch12 = {
            k: (
                batch1[k]
                if "nodes" in k or "weights" in k or k == "problem"
                else torch.cat((batch1[k], batch2[k]), dim=0)
            )
            for k in batch1.keys()
        }
        for name, model in models:
            with self.subTest(name):
                with torch.no_grad():
                    out1 = model(batch1)
                    out2 = model(batch2)
                    out12 = model(batch12)

                self.assertLess(relerr(out12[:2], out1), 1e-4)
                self.assertLess(relerr(out12[2:], out2), 1e-4)


class SGATest(unittest.TestCase):
    def test_sga_diffusion(self):
        torch.manual_seed(1234)
        models = make_models("diffusion", "bspline", False) + make_models(
            "diffusion", "bspline", True
        )
        # Make random input data, but on a consistent quadrature grid
        quad_x = get_quadrature("t12")
        quad_y = get_quadrature("t12")
        batch = make_random_batch("diffusion", 13, quad_x, quad_y)
        batch_x = mirror_x(batch)
        batch_y = mirror_y(batch)
        batch_xy = mirror_y(batch_x)
        for name, model in models:
            with self.subTest(name):
                with torch.no_grad():
                    out = model(batch)
                    out_x = model(batch_x).flip(1)
                    out_y = model(batch_y).flip(2)
                    out_xy = model(batch_xy).flip(1, 2)

                self.assertLess(relerr(out_x, out), 1e-4)
                self.assertLess(relerr(out_y, out), 1e-4)
                self.assertLess(relerr(out_xy, out), 1e-4)

    def test_sga_advdiff(self):
        torch.manual_seed(1234)
        prob, sga = "advection-diffusion", "bspline"
        models = make_models(prob, sga, False) + make_models(prob, sga, True)
        # Make random input data, but on a consistent quadrature grid
        quad_x = get_quadrature("t12")
        quad_y = get_quadrature("t12")
        batch = make_random_batch(prob, 13, quad_x, quad_y)
        batch_x = mirror_x(batch)
        batch_y = mirror_y(batch)
        batch_xy = mirror_y(batch_x)
        for name, model in models:
            with self.subTest(name):
                with torch.no_grad():
                    out = model(batch)
                    out_x = model(batch_x).flip(1)
                    out_y = model(batch_y).flip(2)
                    out_xy = model(batch_xy).flip(1, 2)

                self.assertLess(relerr(out_x, out), 1e-4)
                self.assertLess(relerr(out_y, out), 1e-4)
                self.assertLess(relerr(out_xy, out), 1e-4)


class ScalingTest(unittest.TestCase):
    def test_scaling_diffusion(self):
        torch.manual_seed(1234)
        prob = "diffusion"
        models = make_models(prob, None, True) + make_models(prob, "bspline", True)
        # Make random input data, but on a consistent quadrature grid
        quad_x = get_quadrature("t12")
        quad_y = get_quadrature("m12")
        batch = make_random_batch("diffusion", 13, quad_x, quad_y)
        r = 10.0
        scaled_params = scale_params(batch, r)
        scaled_rhs = scale_rhs(batch, r)
        for name, model in models:
            with self.subTest(name):
                with torch.no_grad():
                    out = model(batch)
                    out_scaledparams = model(scaled_params) * r
                    out_scaledrhs = model(scaled_rhs) / r

                self.assertLess(relerr(out_scaledparams, out), 1e-4)
                self.assertLess(relerr(out_scaledrhs, out), 1e-4)

    def test_scaling_advdiff(self):
        torch.manual_seed(1234)
        prob = "advection-diffusion"
        models = make_models(prob, None, True) + make_models(prob, "bspline", True)
        # Make random input data, but on a consistent quadrature grid
        quad_x = get_quadrature("t12")
        quad_y = get_quadrature("m12")
        batch = make_random_batch("advection-diffusion", 13, quad_x, quad_y)
        r = 10.0
        scaled_params = scale_params(batch, r)
        scaled_rhs = scale_rhs(batch, r)
        for name, model in models:
            with self.subTest(name):
                with torch.no_grad():
                    out = model(batch)
                    out_scaledparams = model(scaled_params) * r
                    out_scaledrhs = model(scaled_rhs) / r

                self.assertLess(relerr(out_scaledparams, out), 1e-4)
                self.assertLess(relerr(out_scaledrhs, out), 1e-4)

    def test_scaling_helmholtz(self):
        torch.manual_seed(1234)
        prob = "helmholtz"
        models = make_models(prob, None, True) + make_models(prob, "bspline", True)
        # Make random input data, but on a consistent quadrature grid
        quad_x = get_quadrature("t12")
        quad_y = get_quadrature("m12")
        batch = make_random_batch("helmholtz", 13, quad_x, quad_y)
        r = 10.0
        scaled_rhs = scale_rhs(batch, r)
        for name, model in models:
            with self.subTest(name):
                with torch.no_grad():
                    out = model(batch)
                    out_scaledrhs = model(scaled_rhs) / r

                self.assertLess(relerr(out_scaledrhs, out), 1e-4)
