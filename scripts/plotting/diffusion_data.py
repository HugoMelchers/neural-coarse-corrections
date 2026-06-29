# Plot a few example problems from a dataset (i.e. theta, f, eta, and u)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from .styles import *

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src import *
from src.utils.quadrature import midpoint
from src.plotting import heatmap, diffmap
import math

Q = 200


def ticks_for(dat):
    lo = dat.min()
    hi = dat.max()
    if lo * hi < 0:
        return [lo, 0.0, hi]
    else:
        return [lo, hi]


def coarse_ticks_for(dat):
    sigdigits = 2

    def rounddown(x):
        ndigits = math.ceil(math.log10(abs(x)))
        mult = 10 ** max(0, ndigits - sigdigits)
        return math.floor(x / mult) * mult

    if dat.dtype in [np.complex64, np.complex128, torch.complex64, torch.complex128]:
        lo = min(dat.real.min(), dat.imag.min())
        hi = max(dat.real.max(), dat.imag.max())
    else:
        lo = dat.min()
        hi = dat.max()

    lo = -rounddown(-lo)
    hi = rounddown(hi)
    if lo * hi < 0:
        return [lo, 0.0, hi]
    else:
        return [lo, hi]


def plot_data(dataname, filename, use_coarse):
    quad = midpoint(Q)
    data = DataSet.load(dataname)
    data = discretise(data, quad, device="cpu")

    fig = plt.figure(figsize=(9, 4))

    gs = GridSpec(2, 3, figure=fig, width_ratios=[3, 2.5, 3.5])
    ax1 = fig.add_subplot(gs[0, 0])  # theta
    ax2 = fig.add_subplot(gs[1, 0])  # f
    ax3 = fig.add_subplot(gs[0, 1])  # g
    ax4 = fig.add_subplot(gs[1, 1])  # eta
    ax5 = fig.add_subplot(gs[:, 2:])  # u
    for ax in [ax1, ax3, ax5]:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))

    for ax in [ax2, ax4]:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f" if use_coarse else "%.2g"))

    theta = data.data["theta"][0]
    f = data.data["f"][0]
    g = data.data["dirichlet:g"][0]
    g_left = g[0]
    g_right = g[1]
    eta = data.data["neumann:eta"][0]
    eta_bot = eta[:, 0]
    eta_top = eta[:, 1]
    u = data.data["u"][0]

    for ax, dat, title in zip([ax1, ax2, ax5], [theta, f, u], [r"$\theta$", r"$f$", r"$u$"]):
        im = ax.imshow(dat.flip(1).T, extent=[0, 1, 0, 1])
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_title(title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax)
        coarse = use_coarse and ax != ax1
        fmt = "%.0f" if coarse else "%.2g"
        cbar.formatter = FormatStrFormatter(fmt)
        cbar.set_ticks(coarse_ticks_for(dat) if coarse else ticks_for(dat))

    ax = ax3
    ax.plot(quad[0], g_left, label="Left")
    ax.plot(quad[0], g_right, label="Right")
    ax.legend()
    ax.set_xticks([0, 1])
    ax.set_yticks(ticks_for(g))
    ax.set_title(r"$g$")

    ax = ax4
    ax.plot(quad[0], eta_bot, label="Bottom")
    ax.plot(quad[0], eta_top, label="Top")
    ax.legend()
    ax.set_xticks([0, 1])
    ax.set_yticks(ticks_for(eta))
    ax.set_title(r"$\eta$")

    plt.tight_layout()
    plt.savefig(filename, transparent=True)
    plt.close()


plot_data("diffusion-dataset-1:0-1", "plots/diffusion/data-sample-set1.pdf", False)
plot_data("diffusion-dataset-2:0-1", "plots/diffusion/data-sample-set2.pdf", False)
plot_data("diffusion-dataset-3:0-1", "plots/diffusion/data-sample-set3.pdf", True)
