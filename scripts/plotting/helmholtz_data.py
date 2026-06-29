# Plot a few example problems from a dataset (i.e. theta, f, eta, and u)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src import *
from src.utils.quadrature import midpoint
from src.plotting import heatmap, diffmap
from .styles import *
import math

Q = 200
datasets = ["helmholtz:0-1"]
filenames = ["data-sample"]


def ticks_for(dat):
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


for dataset, filename in zip(datasets, filenames):
    quad = midpoint(Q)
    data = DataSet.load(f"{dataset}")
    data = discretise(data, quad, device="cpu")

    fig = plt.figure(figsize=(9, 3))

    gs = GridSpec(2, 5, figure=fig)
    ax_k = fig.add_subplot(gs[:, 0])
    ax_f_re = fig.add_subplot(gs[0, 1])
    ax_f_im = fig.add_subplot(gs[1, 1])
    ax_g1 = fig.add_subplot(gs[0, 2])
    ax_g2 = fig.add_subplot(gs[1, 2])
    ax_eta1 = fig.add_subplot(gs[0, 3])
    ax_eta2 = fig.add_subplot(gs[1, 3])
    ax_u_re = fig.add_subplot(gs[0, 4])
    ax_u_im = fig.add_subplot(gs[1, 4])

    for ax in [ax_k, ax_f_re, ax_f_im, ax_g1, ax_g2, ax_eta1, ax_eta2, ax_u_re, ax_u_im]:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    theta = data.data["theta"][0]
    f = data.data["f"][0]
    g = data.data["dirichlet:g"][0]
    g_left = g[0]
    g_right = g[1]
    eta = data.data["neumann:eta"][0]
    eta_bot = eta[:, 0]
    eta_top = eta[:, 1]
    u = data.data["u"][0]

    for ax, dat, title in zip(
        [ax_k, ax_f_re, ax_f_im, ax_u_re, ax_u_im],
        [theta, f.real, f.imag, u.real, u.imag],
        [
            r"$k$",
            r"$\mathrm{Re}(f)$",
            r"$\mathrm{Im}(f)$",
            r"$\mathrm{Re}(u)$",
            r"$\mathrm{Im}(u)$",
        ],
    ):
        im = ax.imshow(dat.flip(1).T, extent=[0, 1, 0, 1])
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_title(title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax)
        cbar.formatter = FormatStrFormatter("%.0f")
        cbar.set_ticks(ticks_for(dat))

    ax_g1.plot(quad[0], g_left.real, label="Real", color="C0")
    ax_g1.plot(quad[0], g_left.imag, label="Imag", color="C0", linestyle="--")
    # ax_g1.legend()
    ax_g1.set_xticks([0, 1])
    ax_g1.set_yticks(ticks_for(g))
    ax_g1.set_title(r"$g$ (left)")

    ax_g2.plot(quad[0], g_right.real, color="C0")
    ax_g2.plot(quad[0], g_right.imag, color="C0", linestyle="--")
    ax_g2.set_xticks([0, 1])
    ax_g2.set_yticks(ticks_for(g))
    ax_g2.set_title(r"$g$ (right)")

    ax_eta1.plot(quad[0], eta_bot.real, label="Real", color="C1")
    ax_eta1.plot(quad[0], eta_bot.imag, label="Imag", color="C1", linestyle="--")
    # ax_eta1.legend()
    ax_eta1.set_xticks([0, 1])
    ax_eta1.set_yticks(ticks_for(eta))
    ax_eta1.set_title(r"$\eta$ (bottom)")

    ax_eta2.plot(quad[0], eta_top.real, color="C1")
    ax_eta2.plot(quad[0], eta_top.imag, color="C1", linestyle="--")
    ax_eta2.set_xticks([0, 1])
    ax_eta2.set_yticks(ticks_for(eta))
    ax_eta2.set_title(r"$\eta$ (top)")

    plt.subplots_adjust(wspace=1, hspace=1)
    plt.tight_layout()
    plt.savefig(f"plots/helmholtz/{filename}.pdf", transparent=True, bbox_inches="tight")
    plt.close()
