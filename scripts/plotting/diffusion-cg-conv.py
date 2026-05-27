from src import *
from src.preconditioning.assemble import assemble
from src.preconditioning.test_preconditioning import make_preconditioned_system
from tqdm import trange
from scipy.sparse.linalg import LinearOperator, aslinearoperator
import matplotlib.pyplot as plt
from .styles import colors, alphas

n = 80
M = "G_diffusion_0.05_1_1.5"
ref = "output/projection/model.pt"
deeponet = "output/diffusion-dataset-3/deeponet/model.pt"
rino = "output/diffusion-dataset-3/rino/model.pt"
varmion = "output/diffusion-dataset-3/varmion/model.pt"
ngo = "output/diffusion-dataset-3/ngo/model.pt"

path = "hist-diffusion-cg-adef2"

# 1. Load one example from the data set
data = DataSet.load("diffusion-dataset-3:17-18")
# 2. Set theta to constant 1. For a B-spline basis this is equivalent to setting the coefficients of
#    the basis to 1.
# data.data["theta"].weights.fill_(1.0)
batch = data.get_batch(slice(0, 1), "cpu")
# 3. Assemble
defaults = {
    "kind": "A-DEF2",
    "side": "inner",
    "with_x0": False,
    "seed": 1234,
    "solver": "cg",
    "solver_kwargs": {"maxiter": 5000},
    "noleftright": True,
    "grid_size": (n + 1, n),
    "tol": 1e-10,
    "alpha": 1.0,
    "random-rhs": True,
}
configs = [
    {
        "name": "A-DEF2 exact",
        "model_path": ref,
        "M": M,
        "color": "black",
        **defaults,
    },
    {
        "name": "RINO",
        "model_path": rino,
        "M": M,
        "color": "C0",
        **defaults,
    },
    {
        "name": "NGO",
        "model_path": ngo,
        "M": M,
        "color": "C1",
        **defaults,
    },
    {
        "name": "NGO (sym.)",
        "model_path": ngo,
        "M": M,
        "symmetrise": True,
        "color": "C2",
        **defaults,
    },
]

try:
    histories = torch.load("scripts/plotting/cache/" + path + ".pt", weights_only=False)
except:
    histories = []
    for config in configs:
        config["model"] = torch.load(config["model_path"], weights_only=False)
        A, b, quad, (ix, iy) = assemble(batch, n + 1, n)
        system = make_preconditioned_system(A, b, batch, config["model"], config, quad, (ix, iy))
        system.kwargs["rtol"] = 0.0
        system.kwargs["atol"] = config["tol"] * system.norm_b
        system.solve()
        niter = len(system.log) - 1
        print(f"{config['name']}: {niter} its, {system.t} s")
        histories.append(system.log)

    torch.save(histories, "scripts/plotting/cache/" + path + ".pt")


plt.figure(figsize=(5.0, 3.5))
for hist, config in zip(histories, configs):
    plt.plot(
        hist,
        label=config["name"],
        color=config["color"],
        alpha=config["alpha"],
    )

plt.xlim(-1, 51)
plt.xlabel("PCG iterations")
plt.ylabel("Relative residual norm")
plt.legend(loc="upper right")
plt.axhline(y=defaults["tol"], color="gray", alpha=0.5, linestyle="--")
plt.yscale("log")
plt.tight_layout()
plt.savefig("plots/diffusion/conv-hist-n80-cg.pdf", transparent=True)
plt.close()
