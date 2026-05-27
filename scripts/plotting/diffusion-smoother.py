from src.preconditioning.smoothers import DiffusionSmoother
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from .styles import *

h = 0.0005
w = 100
H = w * h
theta = np.full((1, 1), 1.0)
t = np.linspace(-w * h, w * h, 2 * w + 1)
X, Y = np.meshgrid(t, t)
zmax = 2.0

smoother = DiffusionSmoother(h, w, theta, 1.5)
filt = smoother.filt / h**2
zmax = min(zmax, np.max(filt))
filt[filt > zmax] = np.nan
filt[filt == 0] = np.nan

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_zlim(0, zmax)
ax.plot_surface(X, Y, filt, cmap=cm.coolwarm)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
ax.xaxis.set_ticks([-H, 0, H])
ax.yaxis.set_ticks([-H, 0, H])
plt.tight_layout()
plt.savefig("plots/diffusion/smoother.pdf", transparent=True)
plt.close()
