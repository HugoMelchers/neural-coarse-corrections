from .tables import makeplot
from .styles import colors, alphas, styles

# Plot example
from src import *
from src.plotting import heatmap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
import math


Q = 200
xs = torch.linspace(0, 1, 2 * Q + 1)[1::2]
nodes = TensorGrid(xs)
data = DataSet.load("diffusion-dataset-3-fine-theta:0-1")
th = data.data["theta"](nodes).reshape(Q, Q)

fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(th.flip(1).T, extent=[0, 1, 0, 1])
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax)
fmt = "%.2g"
cbar.formatter = FormatStrFormatter(fmt)
lo = th.min()
hi = th.max()

cbar.set_ticks([lo, hi])

plt.tight_layout()
plt.savefig("plots/diffusion/fine-theta-example.pdf", transparent=True)


ns = [20, 40, 80, 160, 320]
sm = "G_diffusion_0.05_1_1.5"

# fmt: off
pth = f"precondition/diffusion-dataset-3-fine-theta"
# kinds = ["noprec", "smooth", "exact", "deeponet", "rino", "varmion", "ngo"]
kinds = ["noprec", "smooth", "exact", "ngo"]
cfg = {
    "fileses": [
        ("None"         , [f"projection/{pth}/fgmres/n{n}/noprec"        for n in ns]),
        ("Smoother only", [f"projection/{pth}/fgmres/n{n}/{sm}"          for n in ns]),
        ("Exact"        , [f"projection/{pth}/fgmres/n{n}/A-DEF1-{sm}"   for n in ns]),
        # ("DeepONet"     , [f"diffusion-dataset-3/deeponet/{pth}/fgmres/n{n}/A-DEF1-{sm}" for n in ns]),
        # ("RINO"         , [f"diffusion-dataset-3/rino/{pth}/fgmres/n{n}/A-DEF1-{sm}"     for n in ns]),
        # ("VarMiON"      , [f"diffusion-dataset-3/varmion/{pth}/fgmres/n{n}/A-DEF1-{sm}"  for n in ns]),
        ("NGO"          , [f"diffusion-dataset-3/ngo/{pth}/fgmres/n{n}/A-DEF1-{sm}"      for n in ns]),
    ],
    "colours": [colors[kind] for kind in kinds],
    "alphas": [alphas[kind] for kind in kinds],
    "seps": [3],
    "outfile": "diffusion/barchart-FGMRES-dataset-3-fine-theta",
    "solver": "F-GMRES",
    "title": "",
    "loc": "upper center",
}
makeplot(ns, **cfg)
