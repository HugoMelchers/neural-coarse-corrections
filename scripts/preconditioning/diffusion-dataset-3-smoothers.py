# Run preconditioning tests on diffusion with various smoothers
from copy import deepcopy
import warnings

from src.preconditioning.test_preconditioning import test_preconditioning


data = "diffusion-dataset-3:9000-9001"
prec = "A-DEF1"
Ms = ["jacobi", "blockjacobi_9_9", "ichol"]
n, tol = 160, 4e-6

refmodel = "projection"
models = [
    "diffusion-dataset-3/deeponet",
    "diffusion-dataset-3/rino",
    "diffusion-dataset-3/varmion",
    "diffusion-dataset-3/ngo",
]

solver = "fgmres"

configs = []
names = []
defaults = {
    "dataset_name": data,
    "side": "inner",
    "with_x0": False,
    "seed": 1234,
    "solver": solver,
    "solver_kwargs": {"maxiter": 5000},
    "symmetrise": False,
    "noleftright": True,
}

subfolder = None
save_results = True


def add_cfg(model_path, exact, kind, M, grid, tol, lowtol, filename):
    configs.append(
        {
            "model_path": model_path,
            "kind": kind,
            "M": M,
            "grid_size": grid,
            "tol": tol,
            "filename": filename,
            "random-rhs": lowtol,
            **defaults,
        }
    )


dataname = data.replace(":", "-")
lowtol = False
for M in Ms:
    # Only preconditioning with M⁻¹
    filename = f"{dataname}/{solver}/n{n}/{M}"
    add_cfg(refmodel, True, "PREC", M, (n + 1, n), tol, lowtol, filename)

    # Exact preconditioning
    filename = f"{dataname}/{solver}/n{n}/{prec}-{M}"
    add_cfg(refmodel, True, prec, M, (n + 1, n), tol, lowtol, filename)

    for model in models:
        # Inexact preconditioning
        filename = f"{dataname}/{solver}/n{n}/{prec}-{M}"
        add_cfg(model, False, prec, M, (n + 1, n), tol, lowtol, filename)

if subfolder is not None:
    for config in configs:
        config["filename"] = subfolder + "/" + config["filename"]

n_cfg = len(configs)
max_i = len(f"{n_cfg}")

if not save_results:
    warnings.warn("save_results=False, data will be discarded")

for i, config in enumerate(configs):
    i_str = f"{i+1}".rjust(max_i)
    desc = f"({i_str}/{n_cfg})"
    print(config["model_path"], "/", config["filename"], sep="")
    test_preconditioning(config, desc=desc, save_results=save_results, print_summary=True)
