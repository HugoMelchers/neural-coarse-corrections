import matplotlib.pyplot as plt
import numpy as np
from math import nan
import torch
import matplotlib


def heatmap(data):
    """
        heatmap(data)

    Creates a heatmap of the 2D array `data` (either a `np.ndarray` or `torch.Tensor`) that is
    oriented 'logically', i.e. with the x-axis going left to right and the y-axis going bottom to
    top. This assumes that the array `data` is ordered Y, then X, i.e. the data point at `(x_i,
    y_j)` is at `data[j, i]`.
    """
    if type(data) == torch.Tensor:
        data = data.detach().flip(1).T
    elif type(data) == np.ndarray:
        data = np.flip(data, 1).T
    plt.imshow(data, extent=(0, 1, 0, 1))
    plt.colorbar()


def diffmap_colormap_for_lims(lo, hi):
    M = max(abs(lo), abs(hi))
    colormap = matplotlib.colormaps["bwr"]
    lo_c = 0.5 * (1 + lo / M)
    hi_c = 0.5 * (1 + hi / M)
    colors = colormap(np.linspace(lo_c, hi_c, 101))
    newcolormap = matplotlib.colors.LinearSegmentedColormap.from_list("temp-diffmap", colors)
    return newcolormap


def make_diffmap_colormap(data):
    """
        make_diffmap_colormap(data)

    Returns a tuple `(lo, hi, colormap)`, where `lo` and `hi` are the lowest and highest values in
    `data`, and `colormap` is a colormap created specifically for `data`. This colormap is such that
    when `data` with `clims=(lo, hi)` (which is also the default) and this colormap, the resulting
    heatmap will have a blue-white-red gradient with negative values being blue, zero exactly being
    white, and positive values being red.
    """
    if type(data) == torch.Tensor:
        data = np.array(data)

    lo = np.min(data)
    hi = np.max(data)
    return lo, hi, diffmap_colormap_for_lims(lo, hi)


def diffmap(data):
    """
        diffmap(data)

    Similar to `heatmap(data)`, with the exception that the colormap used in the heatmap is
    customised for showing differences between two 2D arrays: the colormap is chosen as a
    blue-white-red gradient, with blue for negative values, white for zero, and red for positive
    values. Specifically, the colormap is always chosen such that the value 0 corresponds to the
    color white exactly.
    """
    if type(data) == torch.Tensor:
        data = data.detach().flip(1).T
        data = np.array(data)
    elif type(data) == np.ndarray:
        data = np.flip(data, 1).T

    _, _, newcolormap = make_diffmap_colormap(data)

    plt.imshow(data, extent=(0, 1, 0, 1), cmap=newcolormap)
    plt.colorbar()


def plot_bspline2d(trunk, *, dpi=None):
    """
        plot_bspline2d(trunk)

    Given a `BSplineBasis2D` `trunk`, plot the corresponding knot mesh.
    """
    with torch.no_grad():
        p_x = trunk.f_x.get_partition()
        p_y = trunk.f_y.get_partition()

    n = p_x.numel()
    fig, ax = plt.subplots(dpi=dpi) if dpi is not None else plt.subplots()
    ax.set_aspect("equal")
    xs = np.zeros(6 * n)
    ys = np.zeros(6 * n)
    xs[: (3 * n)] = p_x.repeat_interleave(3)
    xs[2 : (3 * n) : 3] = nan
    ys[: (3 * n)] = torch.tensor([0, 1, nan]).repeat(n)

    ys[(3 * n) : (6 * n)] = p_y.repeat_interleave(3)
    ys[(3 * n + 2) : (6 * n) : 3] = nan
    xs[(3 * n) : (6 * n)] = torch.tensor([0, 1, nan]).repeat(n)

    plt.plot(xs.ravel(), ys.ravel())


def groupedbar(data, *, upper=None, seps=[], gap=0.25, gsep=1.5, annotate=None, **kwargs):
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    nbars, ngroups = data.shape
    x = np.arange(ngroups)
    width = 1 / (nbars + len(seps) * gap + gsep)
    offsets = width * np.arange(nbars, dtype=np.float64)
    for sep in seps:
        offsets[sep:] += gap * width
    offsets -= (offsets[0] + offsets[-1]) / 2
    for i, row in enumerate(data):
        kwargs_i = {k: v[i] for (k, v) in kwargs.items()}
        offset = offsets[i]
        rects = ax.bar(x + offset, row, width, **kwargs_i)
        # if annotate is not None:
        #     a_row = annotate[i]
        #     labels = ["${}^*$" if entry else "" for entry in a_row]
        #     ax.bar_label(rects, labels=labels, padding=-5, color=kwargs["color"][i])
    if upper is not None:
        for i, row in enumerate(upper):
            kwargs_i = {k: v[i] for (k, v) in kwargs.items()}
            if "label" in kwargs_i:
                del kwargs_i["label"]
            offset = offsets[i]
            xs = np.full(3 * ngroups, np.nan)
            xs[0::3] = x - 0.75 * width / 2
            xs[1::3] = x + 0.75 * width / 2
            ys = np.full(3 * ngroups, np.nan)
            ys[0::3] = row
            ys[1::3] = row
            ax.plot(xs + offset, ys, **kwargs_i)
