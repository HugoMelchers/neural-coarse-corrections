# Make a plot showing three different distributions:
# 1. GRF, length scale 2
# 2. B-Spline, randn coefficients
# 3. B-spline, randn coefficients * G⁻¹
from src import *
import matplotlib.pyplot as plt
from .styles import *

xs = np.linspace(0, 1, 201)

# GRF
l = 0.2
K = np.exp(-np.square(xs[:, None] - xs[None, :]) / l**2)
vals, vecs = np.linalg.eigh(K)
sel = vals > np.max(vals) * 1e-14
basis1 = vecs[:, sel] * np.sqrt(vals[None, sel])

bspline = BSpline.uniform((0, 1), 9, 3, dtype=torch.float64)
Mi = np.array(bspline.inverse_gram_matrix())
basis2 = np.array(bspline(torch.tensor(xs))).T
basis3 = basis2 @ Mi

n1, n2, n3 = basis1.shape[1], basis2.shape[1], basis3.shape[1]
k = 5

np.random.seed(1234)
plt.figure(figsize=(8, 2))
plt.plot(xs, basis1 @ np.random.randn(n1, k), color="C0", alpha=0.6)
plt.tight_layout()
plt.savefig("plots/diffusion/rhss-dataset-1.pdf", transparent=True)
plt.close()

plt.figure(figsize=(8, 2))
plt.plot(xs, basis2 @ np.random.randn(n2, k), color="C1", alpha=0.6)
plt.tight_layout()
plt.savefig("plots/diffusion/rhss-dataset-2.pdf", transparent=True)
plt.close()

plt.figure(figsize=(8, 2))
plt.plot(xs, basis3 @ np.random.randn(n3, k), color="C2", alpha=0.6)
plt.tight_layout()
plt.savefig("plots/diffusion/rhss-dataset-3.pdf", transparent=True)
plt.close()
