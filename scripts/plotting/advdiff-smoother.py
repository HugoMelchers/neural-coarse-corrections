import matplotlib.pyplot as plt
from src import *
from src.plotting import heatmap
from src.preconditioning.assemble import assemble_advection_diffusion
from .styles import *

nq = 12
nv = 4
vscale = 0.1

t = np.linspace(0, 1, nq)
xs = t[:, None]
ys = t[None, :]

tv = np.linspace(0, 1, 2 * nv + 1)[1::2]
xv = tv[:, None]
yv = tv[None, :]

fig, axs = plt.subplots(nrows=2, ncols=2)
for i, (u, v) in enumerate([(1, 1), (-1, 1), (-1, -1), (1, -1)]):
    D = (u * xs + v * ys).ravel()
    P = np.argsort(D)
    Pi = np.zeros(P.size, dtype=np.int64)
    Pi[P] = np.arange(P.size)
    dat = np.reshape(Pi, (nq, nq))
    ax = axs[i // 2, i % 2]
    ax.imshow(np.flip(dat, 1).T, extent=(0, 1, 0, 1))
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.quiver(xv, yv, np.full((nv, nv), u), np.full((nv, nv), v), pivot="middle", scale=1 / vscale)

plt.tight_layout()
plt.savefig("plots/advdiff/smoother.pdf", transparent=True)
plt.close()
