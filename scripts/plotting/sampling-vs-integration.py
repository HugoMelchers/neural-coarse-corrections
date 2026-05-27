# Produce a plot that visualises the difference between sampling the input fields and integrating
# them, in the context of learning a solution operator to a PDE.
import matplotlib.pyplot as plt
import numpy as np
from src import *
from .styles import *

nq = 100
t = torch.linspace(0, 1, 2 * nq + 1)[1::2]
w = 1 / nq

xs = t[:, None]
ys = t[None, :]
G0 = torch.minimum(xs, ys) * (1 - torch.maximum(xs, ys))

# Green's function doubly projected
basis = BSpline.uniform((0, 1), 6, 2)
vals = basis(t)[1:-1, :]
M = w * (vals @ vals.T)
Mi = torch.inverse(M)
coeffs = w * Mi @ vals @ G0 @ vals.T @ Mi * w
G1 = vals.T @ coeffs @ vals

# Output in basis, input sampled
ids = torch.round((nq - 1) * torch.linspace(0, 1, 8)[1:-1]).to(torch.int64)
C = torch.eye(nq)[ids, :]
coeffs = w * Mi @ vals @ G0 @ C.T * w
G2 = vals.T @ coeffs @ C

plt.figure(figsize=(3, 4.5))
plt.imshow(G0, extent=(0, 1, 0, 1))
plt.xticks([0, 1])
plt.yticks([0, 1])
# plt.title("Exact")
plt.tight_layout()
plt.savefig("plots/sampling-vs-integration-1.pdf", bbox_inches="tight")
plt.close()

plt.figure(figsize=(3, 4.5))
plt.imshow(G1, extent=(0, 1, 0, 1))
plt.xticks([0, 1])
plt.yticks([0, 1])
# plt.title("RINO / NGO")
plt.tight_layout()
plt.savefig("plots/sampling-vs-integration-2.pdf", bbox_inches="tight")
plt.close()

plt.figure(figsize=(3, 4.5))
plt.imshow(G2, extent=(0, 1, 0, 1))
plt.xticks([0, 1])
plt.yticks([0, 1])
# plt.title("DeepONet / VarMiON")
plt.tight_layout()
plt.savefig("plots/sampling-vs-integration-3.pdf", bbox_inches="tight")
plt.close()

# fig, axs = plt.subplots(nrows=1, ncols=3, subplot_kw={"xticks": [], "yticks": []}, figsize=(12, 18))
# axs[0].imshow(G0)
# axs[1].imshow(G1)
# axs[2].imshow(G2)
# axs[0].set_title("Exact")
# axs[1].set_title("RI-DeepONet / NGO")
# axs[2].set_title("DeepONet / VarMiON")
# fig.savefig("plots/for-paper/sampling-vs-integration.pdf", bbox_inches="tight")
