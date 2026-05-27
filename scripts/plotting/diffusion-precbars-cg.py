# Create bar charts

from .tables import fmt_table
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import torch
from .styles import colors, alphas

n = 80
ref = "projection"
data = "diffusion-dataset-3-9000-10000"
pth = f"precondition/{data}"
sm = "G_diffusion_0.05_1_1.5"

# fmt: off
files = [
    ("AD exact"        , f"projection/{pth}/cg/n{n}/AD-{sm}"    ),
    ("A-DEF1 exact"    , f"projection/{pth}/cg/n{n}/A-DEF1-{sm}"),
    ("A-DEF2 exact"    , f"projection/{pth}/cg/n{n}/A-DEF2-{sm}"),
    ("BNN exact"       , f"projection/{pth}/cg/n{n}/BNN-{sm}"   ),

    ("AD RINO"         , f"diffusion-dataset-3/rino/{pth}/cg/n{n}/AD-{sm}"          ),
    ("A-DEF1 RINO"     , f"diffusion-dataset-3/rino/{pth}/cg/n{n}/A-DEF1-{sm}"      ),
    ("A-DEF2 RINO"     , f"diffusion-dataset-3/rino/{pth}/cg/n{n}/A-DEF2-{sm}"      ),
    ("BNN RINO"        , f"diffusion-dataset-3/rino/{pth}/cg/n{n}/BNN-{sm}"         ),

    ("AD NGO"          , f"diffusion-dataset-3/ngo/{pth}/cg/n{n}/AD-{sm}"          ),
    ("A-DEF1 NGO"      , f"diffusion-dataset-3/ngo/{pth}/cg/n{n}/A-DEF1-{sm}"      ),
    ("A-DEF2 NGO"      , f"diffusion-dataset-3/ngo/{pth}/cg/n{n}/A-DEF2-{sm}"      ),
    ("BNN NGO"         , f"diffusion-dataset-3/ngo/{pth}/cg/n{n}/BNN-{sm}"         ),

    ("AD NGO (sym)"    , f"diffusion-dataset-3/ngo/{pth}/cg/n{n}/AD-{sm}-sym"      ),
    ("A-DEF1 NGO (sym)", f"diffusion-dataset-3/ngo/{pth}/cg/n{n}/A-DEF1-{sm}-sym"  ),
    ("A-DEF2 NGO (sym)", f"diffusion-dataset-3/ngo/{pth}/cg/n{n}/A-DEF2-{sm}-sym"  ),
    ("BNN NGO (sym)"   , f"diffusion-dataset-3/ngo/{pth}/cg/n{n}/BNN-{sm}-sym"     ),
]
seps = [1]
outfile = "diffusion/barchart-CG"
solver = "CG"
# fmt: on


means1 = np.full(len(files), np.nan)
means2 = np.full(len(files), np.nan)
fails1 = np.full(len(files), -1)
fails2 = np.full(len(files), -1)

for i, (_, file) in enumerate(files):
    path = pathlib.Path(f"output/{file}.pt")
    if path.exists():
        results = torch.load(path, weights_only=False)
        iters = results["iters"] - 1
        means1[i] = iters.mean()
        maxiter = results["config"]["solver_kwargs"]["maxiter"]
        fails1[i] = (iters == maxiter).sum()

for i, (_, file) in enumerate(files):
    path = pathlib.Path(f"output/{file}-lowtol.pt")
    if path.exists():
        results = torch.load(path, weights_only=False)
        iters = results["iters"] - 1
        means2[i] = iters.mean()
        maxiter = results["config"]["solver_kwargs"]["maxiter"]
        fails2[i] = (iters == maxiter).sum()

print("Mean iterations (default tolerance)")
print(fmt_table(np.reshape(means1, (4, 4))))
print()
print("Failures")
print(fmt_table(np.reshape(fails1, (4, 4))))
print()
print("Mean iterations (low tolerance)")
print(fmt_table(np.reshape(means2, (4, 4))))
print()
print("Failures")
print(fmt_table(np.reshape(fails2, (4, 4))))

data, upper = np.reshape(means1, (4, 4)), np.reshape(means2, (4, 4))
gap = 0.25
gsep = 1.5
kwargs = {
    "label": ["Exact", "RINO", "NGO", "NGO (sym.)"],
    "color": ["black", "C0", "C1", "C2"],
    "alpha": [0.7, 0.7, 0.7, 0.7],
}

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

plt.xticks(np.arange(4), ["AD", "A-DEF1", "A-DEF2", "BNN"])
plt.yscale("log")
plt.ylabel(f"Mean {solver} iterations")
plt.legend(loc="upper left")
plt.tight_layout()
# plt.title(f"{solver} preconditioner effectiveness ($n = {n}$)")
plt.savefig(f"plots/{outfile}.pdf", transparent=True)
plt.close()
