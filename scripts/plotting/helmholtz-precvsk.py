from src import *
import matplotlib.pyplot as plt
from .styles import colors, alphas

n = 80
solver = "fgmres"
smoother = "G_helmholtz_0.05_1_1.5"

solver_displayname = "F-GMRES"
data = "helmholtz:9000-10000"
dataset = DataSet.load(data)
nq = 100
t = torch.linspace(0, 1, 2 * nq + 1)[1::2]
nodes = TensorGrid(t)
k_avg = dataset.data["theta"](nodes).mean(1)
del dataset

dataname = data.replace(":", "-")

# prefix = f"output/projection/precondition/{dataname}/{solver}/n{n}/noprec"
# its0 = torch.load(prefix + ".pt", weights_only=False)["iters"]

prefix = f"output/projection/precondition/{dataname}/{solver}/n{n}/{smoother}"
its1 = torch.load(prefix + ".pt", weights_only=False)["iters"]

prefix = f"output/projection/precondition/{dataname}/{solver}/n{n}/A-DEF1-{smoother}"
its2 = torch.load(prefix + ".pt", weights_only=False)["iters"]

# prefix = (
#     f"output/helmholtz/deeponet/precondition/{dataname}/{solver}/n{n}/A-DEF1-{smoother}"
# )
# its3 = torch.load(prefix + ".pt", weights_only=False)["iters"]

prefix = f"output/helmholtz/rino/precondition/{dataname}/{solver}/n{n}/A-DEF1-{smoother}"
its4 = torch.load(prefix + ".pt", weights_only=False)["iters"]

prefix = f"output/helmholtz/varmion/precondition/{dataname}/{solver}/n{n}/A-DEF1-{smoother}"
its5 = torch.load(prefix + ".pt", weights_only=False)["iters"]

prefix = f"output/helmholtz/ngo/precondition/{dataname}/{solver}/n{n}/A-DEF1-{smoother}"
its6 = torch.load(prefix + ".pt", weights_only=False)["iters"]

plt.figure(figsize=(5.0, 3.5))
# plt.scatter(k_avg, its1 - 1, 0.0, label="None", c=colors['noprec'], alpha=alphas['noprec'])
plt.scatter(k_avg, its1 - 1, 1.0, label="Smoother only", c=colors["smooth"], alpha=alphas["smooth"])
plt.scatter(k_avg, its2 - 1, 1.0, label="Exact", c=colors["exact"], alpha=alphas["exact"])
# plt.scatter(k_avg, its3 - 1, 1.0, label="DeepONet", c=colors['deeponet'], alpha=alphas['deeponet'])
plt.scatter(k_avg, its4 - 1, 1.0, label="RINO", c=colors["rino"], alpha=alphas["rino"])
plt.scatter(k_avg, its5 - 1, 1.0, label="VarMiON", c=colors["varmion"], alpha=alphas["varmion"])
plt.scatter(k_avg, its6 - 1, 1.0, label="NGO", c=colors["ngo"], alpha=alphas["ngo"])
plt.xlabel("Average wave number")
plt.ylabel(f"{solver_displayname} iterations")
plt.legend(loc="upper left")
plt.yscale("log")
plt.tight_layout()
plt.savefig(f"plots/helmholtz/k-vs-{solver}-n{n}.pdf", transparent=True)
plt.close()
