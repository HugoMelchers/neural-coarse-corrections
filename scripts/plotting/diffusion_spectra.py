# 1. Discretise the diffusion equation (with constant diffusivity)
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
M = "G_diffusion_0.05_1_1.5"
ref = "output/projection/model.pt"
deeponet = "output/diffusion-dataset-3/deeponet/model.pt"
rino = "output/diffusion-dataset-3/rino/model.pt"
varmion = "output/diffusion-dataset-3/varmion/model.pt"
ngo = "output/diffusion-dataset-3/ngo/model.pt"
path = "spectra-diffusion-n40"

# 1. Load one example from the data set
data = DataSet.load("diffusion-dataset-3:0-1")
# 2. Set theta to constant 1. For a B-spline basis this is equivalent to setting the coefficients of
#    the basis to 1.
data.data["theta"].weights.fill_(1.0)
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
    "grid_size": (n + 1, n),
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
        "name": "G smoothing",
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
    (evses, comms, nus) = torch.load("scripts/plotting/cache/" + path + ".pt", weights_only=False)
except:
    evses = []
    comms = []
    nus = []
    for config in configs:
        config["model"] = torch.load(config["model_path"], weights_only=False)
        A, b, quad, (ix, iy) = assemble(batch, n + 1, n, config)
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
        # Commutator error
        AAT = AM_dense @ AM_dense.T
        ATA = AM_dense.T @ AM_dense
        comm = np.linalg.norm(AAT - ATA) / np.linalg.norm(AAT)
        # Henrici departure from normality ν
        nu = np.sqrt(np.linalg.norm(AM_dense) ** 2 - np.sum(np.abs(evs) ** 2))
        evses.append(evs)
        comms.append(comm)
        print(f"nu = {nu}, commutator error = {comm}")
        nus.append(nu)

    torch.save((evses, comms, nus), "scripts/plotting/cache/" + path + ".pt")


# Make plot with a subfigure for each spectrum
fig, axs = plt.subplots(nrows=2, ncols=3)
for ax in axs.ravel():
    ax.axhline(y=0, c="grey", linestyle="-", linewidth=0.5, zorder=-1e20)
    ax.axvline(x=0, c="grey", linestyle="-", linewidth=0.5, zorder=-1e20)


axs[1, 0].set_xlabel(r"$\mathrm{Re}(\lambda)$")
axs[1, 0].set_ylabel(r"$\mathrm{Im}(\lambda)$")

alpha = 0.5
axs[0, 0].scatter(evses[1].real, evses[1].imag, 1.0, color="black", alpha=alpha, zorder=1)
axs[1, 0].scatter(evses[2].real, evses[2].imag, 1.0, color="C0", alpha=alpha, zorder=1)
axs[0, 1].scatter(evses[3].real, evses[3].imag, 1.0, color="C1", alpha=alpha, zorder=1)
axs[1, 1].scatter(evses[4].real, evses[4].imag, 1.0, color="C2", alpha=alpha, zorder=1)
axs[0, 2].scatter(evses[5].real, evses[5].imag, 1.0, color="C3", alpha=alpha, zorder=1)
axs[1, 2].scatter(evses[6].real, evses[6].imag, 1.0, color="C4", alpha=alpha, zorder=1)
axs[0, 0].set_title(rf"Smoother only (${comms[1]:.3f}$)")
axs[1, 0].set_title(rf"Exact (${comms[2]:.3f}$)")
axs[0, 1].set_title(rf"DeepONet (${comms[3]:.3f}$)")
axs[1, 1].set_title(rf"RINO (${comms[4]:.3f}$)")
axs[0, 2].set_title(rf"VarMiON (${comms[5]:.3f}$)")
axs[1, 2].set_title(rf"NGO (${comms[6]:.3f}$)")
# plt.suptitle(r"Preconditioned spectra ($\theta \equiv 1$)")
plt.tight_layout()
plt.savefig("plots/diffusion/spectra.pdf", transparent=True)
plt.close()
