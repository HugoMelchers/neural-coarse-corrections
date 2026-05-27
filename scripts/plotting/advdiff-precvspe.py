from src import *
import matplotlib.pyplot as plt
from .styles import colors, alphas

n = 80
solver = "fgmres"
smoother = "gs4"


def scatter(x, y, ymax, *, color, label, alpha=1.0):
    did_converge = y < ymax
    plt.scatter(x[did_converge], y[did_converge], 1.0, color=color, label=label, alpha=alpha)
    if not np.all(did_converge):
        plt.scatter(
            x[~did_converge],
            y[~did_converge],
            10,
            marker="x",
            color=color,
            label=None,
        )


solver_displayname = "F-GMRES"
# data = "advdiff-for-preconditioning:9000-10000"
data = "advdiff-high-peclet:9000-10000"
dataset = DataSet.load(data)
dataset_d = discretise(dataset, "m2", device="cpu")

vx = np.array(dataset_d.data["v_x"][:, 0, 0])
vy = np.array(dataset_d.data["v_y"][:, 0, 0])
pe = np.hypot(vx, vy) / 0.1

dataname = data.replace(":", "-")
prefix = f"output/projection/precondition/{dataname}/{solver}/n{n}/noprec"
its0 = torch.load(prefix + ".pt", weights_only=False)["iters"]

prefix = f"output/projection/precondition/{dataname}/{solver}/n{n}/{smoother}"
its1 = torch.load(prefix + ".pt", weights_only=False)["iters"]

prefix = f"output/projection/precondition/{dataname}/{solver}/n{n}/A-DEF1-{smoother}"
its2 = torch.load(prefix + ".pt", weights_only=False)["iters"]

prefix = f"output/advdiff/deeponet/precondition/{dataname}/{solver}/n{n}/A-DEF1-{smoother}"
its3 = torch.load(prefix + ".pt", weights_only=False)["iters"]

prefix = f"output/advdiff/rino/precondition/{dataname}/{solver}/n{n}/A-DEF1-{smoother}"
its4 = torch.load(prefix + ".pt", weights_only=False)["iters"]

prefix = f"output/advdiff/varmion/precondition/{dataname}/{solver}/n{n}/A-DEF1-{smoother}"
its5 = torch.load(prefix + ".pt", weights_only=False)["iters"]

prefix = f"output/advdiff/ngo/precondition/{dataname}/{solver}/n{n}/A-DEF1-{smoother}"
its6 = torch.load(prefix + ".pt", weights_only=False)["iters"]

plt.figure(figsize=(5.0, 3.5))
plt.scatter(pe, its1 - 1, 1.0, label="Smoother only", c=colors["smooth"], alpha=alphas["smooth"])
plt.scatter(pe, its2 - 1, 1.0, label="Exact", c=colors["exact"], alpha=alphas["exact"])
plt.scatter(pe, its3 - 1, 1.0, label="DeepONet", c=colors["deeponet"], alpha=alphas["deeponet"])
plt.scatter(pe, its4 - 1, 1.0, label="RINO", c=colors["rino"], alpha=alphas["rino"])
plt.scatter(pe, its5 - 1, 1.0, label="VarMiON", c=colors["varmion"], alpha=alphas["varmion"])
plt.scatter(pe, its6 - 1, 1.0, label="NGO", c=colors["ngo"], alpha=alphas["ngo"])
plt.xlabel("Péclet number")
plt.ylabel(f"{solver_displayname} iterations")
plt.legend(loc="upper right")
plt.yscale("log")
plt.tight_layout()
plt.savefig(f"plots/advdiff/peclet-vs-{solver}-n{n}.pdf", transparent=True)
plt.close()
