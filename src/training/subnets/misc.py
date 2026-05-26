import math

import torch
from torch import nn


class ComplexComponentWise(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return torch.view_as_complex(self.inner(torch.view_as_real(x)))


class ReshapeLayer(nn.Module):
    def __init__(self, *new_size):
        """
            ReshapeLayer(*new_size)

        Creates a layer that reshapes its input to the given size. The reshaping ignores the first index (which is assumed to be the samples). For example, ReshapeLayer(5, 7) will reshape a 100×35 input to a 100×5×7 output.
        """
        super(ReshapeLayer, self).__init__()
        self.new_size = new_size

    def forward(self, x):
        batch_size = x.size(0)
        return x.reshape(batch_size, *self.new_size)

    def __str__(self):
        return f"ReshapeLayer{self.new_size}"

    def __repr__(self):
        return f"ReshapeLayer{self.new_size}"


class StrictReshapeLayer(nn.Module):
    def __init__(self, old_size, new_size):
        super(StrictReshapeLayer, self).__init__()
        self.old_size = old_size
        self.new_size = new_size
        old_numel = 1
        for s in old_size:
            old_numel *= s

        new_numel = 1
        for s in new_size:
            new_numel *= s

        assert old_numel == new_numel

    def forward(self, x):
        batch_size = x.size(0)
        assert (
            x.shape[1:] == self.old_size
        ), f"StrictReshapeLayer: expected shape {self.old_size} (ignoring batch index), got shape {x.shape[1:]}"
        return x.reshape(batch_size, *self.new_size)


class ToComplexLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert (
            x.shape[1] == 2
        ), f"ToComplexLayer: expected tensor with size 2 in second dimension, got shape {x.shape}"
        return torch.complex(x[:, 0], x[:, 1])

    def __str__(self):
        return "ToComplexLayer()"

    def __repr__(self):
        return "ToComplexLayer()"


class Const(nn.Module):
    def __init__(self, value):
        """
        Make a layer that always returns the same value, regardless of what arguments it is given.

        I use this for disabling skip connections in UNets, since layer(x) + skip(x) reduces to just layer(x) if skip(x) always returns False.
        """
        super().__init__()
        self.value = value

    def forward(self, *args, **kwargs):
        return self.value

    def __repr__(self):
        return f"Const({self.value})"


class MatrixInverse(nn.Module):
    def __init__(self):
        """
        Create a Module that accepts (a batch of) square matrices and returns their inverses.
        """
        super().__init__()

    def forward(self, matrices):
        return matrices.inverse()


class BoundedActivation(nn.Module):
    def __init__(self, activation, lo, hi, log):
        super().__init__()
        bounds = activation_bounds(activation)
        if bounds is None:
            raise ValueError(f"output bounds not known for activation {activation}")

        a, b = bounds
        self.activation = get_activation(activation)
        self.log = log
        if self.log:
            # the activation function maps to (a, b)
            # so take an affine map to move this to (log lo, log hi)
            loglo = math.log(lo)
            loghi = math.log(hi)
            self.scale = (loghi - loglo) / (b - a)
            self.shift = loglo - self.scale * a
        else:
            # the activation function maps to (a, b)
            # so take an affine map (scale + shift) to move this to (lo, hi)
            self.scale = (hi - lo) / (b - a)
            self.shift = lo - self.scale * a
            # Now:
            #   scale * a + shift
            #   = (hi - lo) / (b - a) * a + lo - (hi - lo) / (b - a) * a
            #   = lo
            # and
            #   scale * b + shift
            #   = (hi - lo) / (b - a) * b + lo - (hi - lo) / (b - a) * a
            #   = (hi - lo) / (b - a) * (b - a) + lo
            #   = hi

    def forward(self, x):
        out = self.scale * self.activation(x) + self.shift
        if self.log:
            return out.exp()
        else:
            return out


def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "leakyrelu":
        return nn.LeakyReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "hardsigmoid":
        return nn.Hardsigmoid()
    elif name == "hardtanh":
        return nn.Hardtanh()
    elif name == "relu6":
        return nn.ReLU6()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "softsign":
        return nn.Softsign()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "identity" or name is None:
        return nn.Identity()
    else:
        raise ValueError(f'unknown activation function "{name}"')


def activation_bounds(name):
    if name == "hardsigmoid" or name == "sigmoid":
        return (0.0, 1.0)
    elif name == "hardtanh" or name == "tanh" or name == "softsign":
        return (-1.0, 1.0)
    elif name == "relu6":
        return (0.0, 6.0)
    else:
        return None


def dense_net(sizes, activation=nn.LeakyReLU()):
    layers = []
    for i in range(0, len(sizes) - 1):
        if i != 0:
            layers.append(activation)
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    return nn.Sequential(*layers)


def conv_net(channels, widths, strides, with_bn=False, activation=nn.LeakyReLU()):
    layers = []
    if type(with_bn) == bool:
        with_bn = [with_bn for _ in widths]
    for i in range(len(channels) - 1):
        layer = nn.Conv2d(channels[i], channels[i + 1], widths[i], strides[i])
        layers.append(layer)
        if i != len(channels) - 2:
            layers.append(activation)
            if with_bn[i]:
                layers.append(nn.BatchNorm2d(channels[i + 1]))

    return nn.Sequential(*layers)


def trconv_net(channels, widths, strides, with_bn=False, activation=nn.LeakyReLU()):
    layers = []
    # with_bn can be a single boolean, or a list to add batch normalisation to some layers
    if type(with_bn) == bool:
        with_bn = [with_bn for _ in widths]
    for i in range(len(channels) - 1):
        layer = nn.ConvTranspose2d(channels[i], channels[i + 1], widths[i], strides[i])
        layers.append(layer)
        if i != len(channels) - 2:
            layers.append(activation)
            if with_bn[i]:
                layers.append(nn.BatchNorm2d(channels[i + 1]))

    return nn.Sequential(*layers)
