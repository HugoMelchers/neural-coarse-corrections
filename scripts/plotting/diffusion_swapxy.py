from .tables import makeplot
from .styles import colors, alphas, styles

import torch
import matplotlib.pyplot as plt


ns = [20, 40, 80, 160, 320]
sm = "G_diffusion_0.05_1_1.5"

# fmt: off
pth1 = f"precondition/diffusion-dataset-3-9000-10000"
pth2 = f"precondition/swapxy/diffusion-dataset-3-9000-10000"
kinds = ["smooth", "exact", "ngo", "ngo"]
colours= ["black", "C0", "C1", "C2"]
cfg = {
    "fileses": [
        ("Smoother only"    , [f"projection/{pth1}/fgmres/n{n}/{sm}"        for n in ns]),
        ("Exact"            , [f"projection/{pth1}/fgmres/n{n}/A-DEF1-{sm}" for n in ns]),
        ("NGO"              , [f"diffusion-dataset-3/ngo/{pth1}/fgmres/n{n}/A-DEF1-{sm}"       for n in ns]),
        ("NGO (swapped BCs)", [f"diffusion-dataset-3/ngo/{pth2}/fgmres/n{n}/A-DEF1-{sm}"       for n in ns]),
    ],
    "colours": colours,
    "alphas": [alphas[kind] for kind in kinds],
    "seps": [2],
    "outfile": "diffusion/barchart-FGMRES-dataset-3-swapxy",
    "solver": "F-GMRES",
    "title": "",
    "loc": "upper center",
}
makeplot(ns, **cfg)

# Example convergence curve
ni = 2 # n = 80

fig = plt.figure(figsize=(5.0, 3.5))
for i in range(4):
    path = cfg['fileses'][i][1][ni]
    path = f"output/{path}.pt"
    log = torch.load(path, weights_only=False)['logs'][0]
    plt.plot(log, label=cfg['fileses'][i][0], color=colours[i])

plt.xlim(-1, 51)
plt.xlabel("F-GMRES iterations")
plt.ylabel("Relative residual norm")
plt.legend(loc="upper right")
plt.axhline(y=1.6e-5, color="gray", alpha=0.5, linestyle="--")
plt.yscale("log")
plt.tight_layout()
plt.savefig("plots/diffusion/conv-hist-n80-swapxy.pdf", transparent=True)
plt.close()
