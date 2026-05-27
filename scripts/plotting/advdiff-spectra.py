# 1. Discretise the advection-diffusion equation (with constant diffusivity)
# 2. Assemble the linear system as well as preconditioned variants
#    - No preconditioning
#    - Smoother only
#    - A-DEF1 exact
#    - A-DEF1 VarMiON
#    - A-DEF1 NGO
# 3. Compare the spectra of different preconditioned systems
from src import *
from src.preconditioning.assemble import assemble
from src.preconditioning.test_preconditioning import make_preconditioned_system
from tqdm import trange
from scipy.sparse.linalg import LinearOperator, aslinearoperator
import matplotlib.pyplot as plt
from .styles import colors, alphas

n = 40
M = "gs4"
ref = "output/projection/model.pt"
deeponet = "output/advdiff/deeponet/model.pt"
rino = "output/advdiff/rino/model.pt"
varmion = "output/advdiff/varmion/model.pt"
ngo = "output/advdiff/ngo/model.pt"
path = "spectra-advdiff-n40"

# 1. Load one example from the data set
data = DataSet.load("advdiff-low-peclet:0-1")
# 2. Set theta to constant 1. For a B-spline basis this is equivalent to setting the coefficients of
#    the basis to 1.
data.data["theta"].weights.fill_(0.1)
data.data["v_x"].weights.fill_(0.6)
data.data["v_y"].weights.fill_(0.8)
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
        "color": colors["noprec"],
        "alpha": alphas["noprec"],
        **defaults,
    },
    {
        "name": "Downwind BGS",
        "model_path": ref,
        "kind": "PREC",
        "M": M,
        "color": colors["smooth"],
        "alpha": alphas["smooth"],
        **defaults,
    },
    {
        "name": "A-DEF1 exact",
        "model_path": ref,
        "kind": "A-DEF1",
        "M": M,
        "color": colors["exact"],
        "alpha": alphas["exact"],
        **defaults,
    },
    {
        "name": "DeepONet",
        "model_path": deeponet,
        "kind": "A-DEF1",
        "M": M,
        "color": colors["deeponet"],
        "alpha": alphas["deeponet"],
        **defaults,
    },
    {
        "name": "RINO",
        "model_path": rino,
        "kind": "A-DEF1",
        "M": M,
        "color": colors["rino"],
        "alpha": alphas["rino"],
        **defaults,
    },
    {
        "name": "VarMiON",
        "model_path": varmion,
        "kind": "A-DEF1",
        "M": M,
        "color": colors["varmion"],
        "alpha": alphas["varmion"],
        **defaults,
    },
    {
        "name": "NGO",
        "model_path": ngo,
        "kind": "A-DEF1",
        "M": M,
        "color": colors["ngo"],
        "alpha": alphas["ngo"],
        **defaults,
    },
]

try:
    evses = torch.load("scripts/plotting/cache/" + path + ".pt", weights_only=False)
except:
    evses = []
    for config in configs:
        config["model"] = torch.load(config["model_path"], weights_only=False)
        A, b, quad, (ix, iy) = assemble(batch, n + 1, n + 1)
        system = make_preconditioned_system(A, b, batch, config["model"], config, quad, (ix, iy))
        AM = aslinearoperator(system.A) @ system.Pr

        N = A.shape[0]
        AM_dense = np.zeros((N, N))
        b = np.zeros(N)
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
# plt.suptitle(r"Preconditioned spectra ($\mathrm{Pe} = 10$)")
plt.tight_layout()
plt.savefig("plots/advdiff/spectra.pdf", transparent=True)
plt.close()
