# Print a LaTeX table containing the mean F-GMRES iterations for the 160x160 diffusion problems,
# using various smoothers

import torch
from pathlib import Path
import numpy as np

n = 160

smoother_codes = [
    "G_diffusion_0.05_1_1.5",
    "ichol",
    "jacobi",
    "blockjacobi_9_9",
]
smoother_names = [
    "Green's function   ",
    "Incomplete Cholesky",
    "Jacobi             ",
    "Block Jacobi       ",
]

pth = f"precondition/diffusion-dataset-3-9000-10000"
kinds = ["smooth", "exact", "varmion", "ngo"]
coarse_correction_names = [
    "None",
    "Exact",
    "DeepONet",
    "RINO",
    "VarMiON",
    "NGO",
]

print(
    rf"""
\begin{{tabular}}{{l|rrrrrr}}
    \toprule
    \multirow{{2}}*{{Smoother}} & \multicolumn{{6}}{{c}}{{Coarse correction}} \\
    & {' & '.join(coarse_correction_names)} \\
    \midrule"""
)
for sm, name in zip(smoother_codes, smoother_names):
    paths = [
        Path(f"output/projection/{pth}/fgmres/n{n}/{sm}.pt"),
        Path(f"output/projection/{pth}/fgmres/n{n}/A-DEF1-{sm}.pt"),
        Path(f"output/diffusion-dataset-3/deeponet/{pth}/fgmres/n{n}/A-DEF1-{sm}.pt"),
        Path(f"output/diffusion-dataset-3/rino/{pth}/fgmres/n{n}/A-DEF1-{sm}.pt"),
        Path(f"output/diffusion-dataset-3/varmion/{pth}/fgmres/n{n}/A-DEF1-{sm}.pt"),
        Path(f"output/diffusion-dataset-3/ngo/{pth}/fgmres/n{n}/A-DEF1-{sm}.pt"),
    ]
    mean_iters = [
        torch.load(path, weights_only=False)["iters"].mean() - 1 if path.exists() else np.nan
        for path in paths
    ]
    print(f"    {name} & " + " & ".join(f"{m:6.1f}" for m in mean_iters) + r" \\")

print(
    r"""    \bottomrule
\end{tabular}
"""
)
