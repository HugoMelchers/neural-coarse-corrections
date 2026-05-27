from .tables import makeplot
from . import styles
import pathlib
import torch
import matplotlib.pyplot as plt
import numpy as np

n = 80
sm = "G_diffusion_0.05_1_1.5"

pth = f"precondition/diffusion-dataset-3-9000-10000"
files = [
    ("None", f"projection/{pth}/fgmres/n{n}/noprec"),
    ("Smoother only", f"projection/{pth}/fgmres/n{n}/{sm}"),
    ("Exact", f"projection/{pth}/fgmres/n{n}/A-DEF1-{sm}"),
    ("DeepONet", f"diffusion-dataset-3/deeponet/{pth}/fgmres/n{n}/A-DEF1-{sm}"),
    ("RINO", f"diffusion-dataset-3/rino/{pth}/fgmres/n{n}/A-DEF1-{sm}"),
    ("VarMiON", f"diffusion-dataset-3/varmion/{pth}/fgmres/n{n}/A-DEF1-{sm}"),
    ("NGO", f"diffusion-dataset-3/ngo/{pth}/fgmres/n{n}/A-DEF1-{sm}"),
]

# Horizontal boxplots are drawn with the first entry at the bottom, which I want to be at the top
# instead, so I reverse the order of everything
files = files[::-1]


iterses = []
for name, file in files:
    path = pathlib.Path(f"output/{file}.pt")
    data = torch.load(path, weights_only=False)
    iters = data["iters"] - 1
    iterses.append(iters)

plt.figure(figsize=(5.0, 3.5))
plt.boxplot(iterses, orientation="horizontal", whis=(5, 95))
plt.yticks(range(1, 1 + len(files)), [name for (name, _) in files])
plt.xscale("log")
plt.xlabel("F-GMRES iterations")
plt.tight_layout()
plt.savefig(f"plots/diffusion/iter-distribution-n{n}.pdf", transparent=True)
plt.close()

sm = "gs4"
pth1 = f"precondition/advdiff-low-peclet-9000-10000"
pth2 = f"precondition/advdiff-low-peclet-9000-10000"
files = [
    ("None", f"projection/{pth1}/fgmres/n{n}/noprec"),
    ("Smoother only", f"projection/{pth2}/fgmres/n{n}/{sm}"),
    ("Exact", f"projection/{pth2}/fgmres/n{n}/A-DEF1-{sm}"),
    ("DeepONet", f"advdiff/deeponet/{pth2}/fgmres/n{n}/A-DEF1-{sm}"),
    ("RINO", f"advdiff/rino/{pth2}/fgmres/n{n}/A-DEF1-{sm}"),
    ("VarMiON", f"advdiff/varmion/{pth2}/fgmres/n{n}/A-DEF1-{sm}"),
    ("NGO", f"advdiff/ngo/{pth2}/fgmres/n{n}/A-DEF1-{sm}"),
]

# Horizontal boxplots are drawn with the first entry at the bottom, which I want to be at the top
# instead, so I reverse the order of everything
files = files[::-1]


iterses = []
for name, file in files:
    path = pathlib.Path(f"output/{file}.pt")
    data = torch.load(path, weights_only=False)
    iters = data["iters"] - 1
    iterses.append(iters)

plt.figure(figsize=(5.0, 3.5))
plt.boxplot(iterses, orientation="horizontal", whis=(5, 95))
plt.yticks(range(1, 1 + len(files)), [name for (name, _) in files])
plt.xscale("log")
plt.xlabel("F-GMRES iterations")
plt.tight_layout()
plt.savefig(f"plots/advdiff/iter-distribution-n{n}-low-peclet.pdf", transparent=True)
plt.close()

sm = "gs4"
pth1 = f"precondition/advdiff-high-peclet-9000-10000"
pth2 = f"precondition/advdiff-high-peclet-9000-10000"
files = [
    ("None", f"projection/{pth1}/fgmres/n{n}/noprec"),
    ("Smoother only", f"projection/{pth2}/fgmres/n{n}/{sm}"),
    ("Exact", f"projection/{pth2}/fgmres/n{n}/A-DEF1-{sm}"),
    ("DeepONet", f"advdiff/deeponet/{pth2}/fgmres/n{n}/A-DEF1-{sm}"),
    ("RINO", f"advdiff/rino/{pth2}/fgmres/n{n}/A-DEF1-{sm}"),
    ("VarMiON", f"advdiff/varmion/{pth2}/fgmres/n{n}/A-DEF1-{sm}"),
    ("NGO", f"advdiff/ngo/{pth2}/fgmres/n{n}/A-DEF1-{sm}"),
]

# Horizontal boxplots are drawn with the first entry at the bottom, which I want to be at the top
# instead, so I reverse the order of everything
files = files[::-1]


iterses = []
for name, file in files:
    path = pathlib.Path(f"output/{file}.pt")
    data = torch.load(path, weights_only=False)
    iters = data["iters"] - 1
    iterses.append(iters)

plt.figure(figsize=(5.0, 3.5))
plt.boxplot(iterses, orientation="horizontal", whis=(5, 95))
plt.yticks(range(1, 1 + len(files)), [name for (name, _) in files])
plt.xscale("log")
plt.xlabel("F-GMRES iterations")
plt.tight_layout()
plt.savefig(f"plots/advdiff/iter-distribution-n{n}-high-peclet.pdf", transparent=True)
plt.close()

sm = "G_helmholtz_0.05_1_1.5"
pth1 = f"precondition/helmholtz-9000-10000"
pth2 = f"precondition/helmholtz-9000-10000"
files = [
    ("None", f"projection/{pth2}/fgmres/n{n}/noprec"),
    ("Smoother only", f"projection/{pth2}/fgmres/n{n}/{sm}"),
    ("Exact", f"projection/{pth2}/fgmres/n{n}/A-DEF1-{sm}"),
    ("DeepONet", f"helmholtz/deeponet/{pth2}/fgmres/n{n}/A-DEF1-{sm}"),
    ("RINO", f"helmholtz/rino/{pth2}/fgmres/n{n}/A-DEF1-{sm}"),
    ("VarMiON", f"helmholtz/varmion/{pth2}/fgmres/n{n}/A-DEF1-{sm}"),
    ("NGO", f"helmholtz/ngo/{pth2}/fgmres/n{n}/A-DEF1-{sm}"),
]

# Horizontal boxplots are drawn with the first entry at the bottom, which I want to be at the top
# instead, so I reverse the order of everything
files = files[::-1]


iterses = []
for name, file in files:
    path = pathlib.Path(f"output/{file}.pt")
    try:
        data = torch.load(path, weights_only=False)
        iters = data["iters"] - 1
        iterses.append(iters)
    except:
        iterses.append([])

plt.figure(figsize=(5.0, 3.5))
plt.boxplot(iterses, orientation="horizontal", whis=(5, 95))
plt.yticks(range(1, 1 + len(files)), [name for (name, _) in files])
plt.xscale("log")
plt.xlabel("F-GMRES iterations")
plt.tight_layout()
plt.savefig(f"plots/helmholtz/iter-distribution-n{n}.pdf", transparent=True)
plt.close()
