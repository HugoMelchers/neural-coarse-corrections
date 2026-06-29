from copy import deepcopy
import warnings

from src.preconditioning.test_preconditioning import test_preconditioning

# data = "advdiff-low-peclet:9000-10000"
data = "advdiff-high-peclet:9000-10000"

prec = "A-DEF1"
M = "gs4"
ns_tols = [
    # (20, 2.56e-4),
    # (40, 6.4e-5),
    (80, 1.6e-5),
    # (160, 4e-6),
    # (320, 1e-6),
]

Ns = [50, 100, 200, 400, 800]
refmodel = None
models = [f"advdiff/less-train/ngo-N{N}" for N in Ns]
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
    "pe_max": 60.0,
}

save_results = True
subfolder = None


def add_cfg(model_path, exact, kind, M, grid, tol, filename):
    configs.append(
        {
            "model_path": model_path,
            "exact": exact,
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
for lowtol in [False, True]:
    for n, tol in ns_tols:
        lt = ""
        if lowtol:
            tol = 1e-10
            lt = "-lowtol"

        for model in models:
            # Inexact preconditioning
            filename = f"{dataname}/{solver}/n{n}/{prec}-{M}{lt}"
            add_cfg(model, False, prec, M, (n + 1, n + 1), tol, filename)

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
