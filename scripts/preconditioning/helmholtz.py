# Run lots of precondition tests
from copy import deepcopy
import warnings

from src.preconditioning.test_preconditioning import test_preconditioning

data = "helmholtz:9000-9001"
precs = ["A-DEF1"]
Ms = ["G_helmholtz_0.05_1_1.5"]
ns_tols = [
    (20, 2.56e-4),
    (40, 6.4e-5),
    (80, 1.6e-5),
    (160, 4e-6),
    (320, 1e-6),
]

refmodel = "projection"
models = ["helmholtz/deeponet", "helmholtz/rino", "helmholtz/varmion", "helmholtz/ngo"]
solver = "fgmres"

configs = []
names = []
defaults = {
    "side": "inner",
    "with_x0": False,
    "seed": 1234,
    "solver": solver,
    "solver_kwargs": {"maxiter": 5000},
    "symmetrise": False,
    "noleftright": True,
    "nobottomtop": False,
}

with_none = True
with_prec = True
with_exact = True
with_inexact = True
subfolder = None
save_results = True


def add_cfg(data, model_path, kind, M, grid, tol, filename, lowtol):
    configs.append(
        {
            "dataset_name": data,
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
for lowtol in [False, True]:
    for n, tol in ns_tols:
        lt = ""
        if lowtol:
            tol = 1e-10
            lt = "-lowtol"

        if with_none:
            # No preconditioning
            filename = f"{dataname}/{solver}/n{n}/noprec{lt}"
            add_cfg(data, refmodel, "PREC", None, (n + 1, n), tol, filename, lowtol)

        for M in Ms:
            if with_prec:
                # Only preconditioning with M⁻¹
                filename = f"{dataname}/{solver}/n{n}/{M}{lt}"
                add_cfg(data, refmodel, "PREC", M, (n + 1, n), tol, filename, lowtol)

            for prec in precs:
                if with_exact:
                    # Exact preconditioning
                    filename = f"{dataname}/{solver}/n{n}/{prec}-{M}{lt}"
                    add_cfg(data, refmodel, prec, M, (n + 1, n), tol, filename, lowtol)

                if with_inexact:
                    for model in models:
                        # Inexact preconditioning
                        filename = f"{dataname}/{solver}/n{n}/{prec}-{M}{lt}"
                        add_cfg(data, model, prec, M, (n + 1, n), tol, filename, lowtol)

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
