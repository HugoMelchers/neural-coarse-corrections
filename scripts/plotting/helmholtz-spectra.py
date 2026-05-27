from src import *
from src.preconditioning.assemble import assemble
from src.preconditioning.test_preconditioning import make_preconditioned_system
from tqdm import trange
from scipy.sparse.linalg import LinearOperator, aslinearoperator
import matplotlib.pyplot as plt
from src.plotting import heatmap
from copy import deepcopy
from .styles import *

n = 40
M = "G_helmholtz_0.05_1_1.5"
ref = "output/projection/model.pt"
deeponet = "output/helmholtz/deeponet/model.pt"
rino = "output/helmholtz/rino/model.pt"
varmion = "output/helmholtz/varmion/model.pt"
ngo = "output/helmholtz/ngo/model.pt"

data = DataSet.load("helmholtz:0-1")
for k in [5, 20]:
    path = f"spectra-helmholtz-k{k}"
    data.data["theta"].weights.fill_(k)
    batch = data.get_batch(slice(0, 1), "cpu")
    # 3. Assemble
    defaults = {
        "side": "right",
        "with_x0": False,
        "seed": 1234,
        "solver": "gmres",
        "solver_kwargs": {"maxiter": 5000},
        "symmetrise": False,
        "noleftright": True,
        "grid_size": (n + 1, n + 1),
        "tol": 1e-10,
    }
    configs = [
        {
            "name": "None",
            "model_path": ref,
            "kind": "PREC",
            "M": None,
            "color": "black",
            "alpha": 1.0,
            **defaults,
        },
        {
            "name": "G smoothing",
            "model_path": ref,
            "kind": "PREC",
            "M": M,
            "color": "black",
            "alpha": 0.8,
            **defaults,
        },
        {
            "name": "A-DEF1 exact",
            "model_path": ref,
            "kind": "A-DEF1",
            "M": M,
            "color": "black",
            "alpha": 0.6,
            **defaults,
        },
        {
            "name": "DeepONet",
            "model_path": deeponet,
            "kind": "A-DEF1",
            "M": M,
            "color": "C0",
            "alpha": 1.0,
            **defaults,
        },
        {
            "name": "RINO",
            "model_path": rino,
            "kind": "A-DEF1",
            "M": M,
            "color": "C1",
            "alpha": 1.0,
            **defaults,
        },
        {
            "name": "VarMiON",
            "model_path": varmion,
            "kind": "A-DEF1",
            "M": M,
            "color": "C2",
            "alpha": 1.0,
            **defaults,
        },
        {
            "name": "NGO",
            "model_path": ngo,
            "kind": "A-DEF1",
            "M": M,
            "color": "C3",
            "alpha": 1.0,
            **defaults,
        },
    ]

    try:
        evses = torch.load("scripts/plotting/cache/" + path + ".pt", weights_only=False)
    except:
        evses = []
        for config in configs:
            config["model"] = torch.load(config["model_path"], weights_only=False)
            A, b, quad, (ix, iy) = assemble(batch, n + 1, n)
            system = make_preconditioned_system(
                A, b, deepcopy(batch), config["model"], config, quad, (ix, iy)
            )
            AM = aslinearoperator(system.A) @ system.Pr
            try:
                Q = system.Pr.Ps[0]
            except:
                Q = None

            N = A.shape[0]
            AM_dense = np.zeros((N, N), dtype=np.complex128)
            b = np.zeros(N, dtype=np.complex128)
            for i in trange(N):
                b[i] = 1.0
                AM_dense[:, i] = AM @ b
                b[i] = 0.0

            evs = np.linalg.eigvals(AM_dense)
            evses.append(evs)

        torch.save(evses, "scripts/plotting/cache/" + path + ".pt")

    # Make plot with a subfigure for each spectrum
    fig, axs = plt.subplots(nrows=2, ncols=3)
    for ax in axs.ravel():
        ax.axhline(y=0, c="grey", linestyle="-", linewidth=0.5, zorder=-1e20)
        ax.axvline(x=0, c="grey", linestyle="-", linewidth=0.5, zorder=-1e20)

    axs[1, 0].set_xlabel(r"$\mathrm{Re}(\lambda)$")
    axs[1, 0].set_ylabel(r"$\mathrm{Im}(\lambda)$")

    alpha = 0.5
    axs[0, 0].scatter(evses[1].real, evses[1].imag, 1.0, color="black", alpha=alpha)
    axs[1, 0].scatter(evses[2].real, evses[2].imag, 1.0, color="C0", alpha=alpha)
    axs[0, 1].scatter(evses[3].real, evses[3].imag, 1.0, color="C1", alpha=alpha)
    axs[1, 1].scatter(evses[4].real, evses[4].imag, 1.0, color="C2", alpha=alpha)
    axs[0, 2].scatter(evses[5].real, evses[5].imag, 1.0, color="C3", alpha=alpha)
    axs[1, 2].scatter(evses[6].real, evses[6].imag, 1.0, color="C4", alpha=alpha)
    axs[0, 0].set_title("Smoother only")
    axs[1, 0].set_title("Exact")
    axs[0, 1].set_title("DeepONet")
    axs[1, 1].set_title("RINO")
    axs[0, 2].set_title("VarMiON")
    axs[1, 2].set_title("NGO")
    # plt.suptitle(f"Preconditioned spectra ($k = {k}$)")
    plt.tight_layout()
    plt.savefig(f"plots/helmholtz/spectra-k{k}.pdf", transparent=True)
    plt.close()
