from src import *
from scipy.sparse.linalg import aslinearoperator, LinearOperator, gmres
from copy import deepcopy
from src.preconditioning.assemble import assemble
from src.preconditioning.test_preconditioning import make_preconditioned_system
import matplotlib.pyplot as plt
from .styles import *
from src.plotting import heatmap, diffmap, make_diffmap_colormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Logging(LinearOperator):
    def __init__(self, inner):
        self.inner = inner
        self.shape = inner.shape
        self.dtype = inner.dtype
        self.log = []

    def _matvec(self, x):
        self.log.append(deepcopy(x))
        return self.inner._matvec(x)


datas = ["diffusion-dataset-1:0-1", "diffusion-dataset-3:0-1"]
filenames = ["gmres-rhss-dataset-1", "gmres-rhss-dataset-3"]
titles = ["Dataset 1", "Dataset 3"]

for data, filename, title in zip(datas, filenames, titles):
    data = DataSet.load(data)
    batch = data.get_batch(slice(0, 1), "cpu")
    M = "G_diffusion_0.05_1_1.5"
    n = 40
    ref = "output/projection/model.pt"
    config = {
        "name": "A-DEF1 exact",
        "model_path": ref,
        "kind": "A-DEF1",
        "M": M,
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

    config["model"] = torch.load(config["model_path"], weights_only=False)
    A, b, quad, (ix, iy) = assemble(batch, n + 1, n)
    system = make_preconditioned_system(A, b, batch, config["model"], config, quad, (ix, iy))
    system.Pr = Logging(aslinearoperator(system.Pr))
    system.solve()

    its = [0, 1, 2, 3]

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(8, 2))
    for i, j in enumerate(its):
        ax = axs[i]
        rhs = system.Pr.log[j]
        dat = np.reshape(rhs, (n - 1, n))
        dat = np.flip(dat, 1).T
        lo, hi, cmap = make_diffmap_colormap(dat[1:-1, 1:-1])
        im = ax.imshow(dat, cmap=cmap, extent=(0, 1, 0, 1), clim=(lo, hi))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Iteration {j + 1}")

        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = plt.colorbar(im, cax)

    # plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"plots/diffusion/{filename}.pdf", transparent=True)
    plt.close()
