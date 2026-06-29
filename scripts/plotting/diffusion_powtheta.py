from .tables import makeplot
from .styles import colors, alphas, styles
from src import *
from pathlib import Path

# First, compute the contrast ratios of theta
dataset = "diffusion-dataset-3:9000-10000"
data = DataSet.load(dataset)
theta = data.data["theta"]
xs = torch.linspace(0, 1, 201)
nodes = TensorGrid(xs)

theta = theta(nodes)
theta_min = theta.min(axis=1).values
theta_max = theta.max(axis=1).values
ratio = np.array(theta_max / theta_min)
del theta
del data

n = 80
pows = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
sm = "G_diffusion_0.05_1_1.5"
dataname = dataset.replace(":", "-") + f"/fgmres/n{n}"

iterses0 = []  # Smoother only
iterses1 = []  # Exact two-level
iterses2 = []  # NGO two-level (ood)
iterses3 = []  # NGO two-level
# Load the baseline results (pow = 1, except that they're not called that)
path = Path(f"output/projection/precondition/{dataname}/{sm}.pt")
iterses0.append(torch.load(path, weights_only=False)["iters"])
path = Path(f"output/projection/precondition/{dataname}/A-DEF1-{sm}.pt")
iterses1.append(torch.load(path, weights_only=False)["iters"])
# Load NGO result (approach 1, if file exists)
path = Path(f"output/diffusion-dataset-3/ngo/precondition/{dataname}/A-DEF1-{sm}.pt")
dat = torch.load(path, weights_only=False)["iters"]
iterses2.append(dat)
iterses3.append(dat)


for pow in pows[1:]:
    # Load reference results
    path = Path(f"output/projection/precondition/powtheta-{pow}-notngo/{dataname}/{sm}.pt")
    iterses0.append(torch.load(path, weights_only=False)["iters"])
    path = Path(f"output/projection/precondition/powtheta-{pow}-notngo/{dataname}/A-DEF1-{sm}.pt")
    iterses1.append(torch.load(path, weights_only=False)["iters"])
    # Load NGO result (approach 1, if file exists)
    path = Path(
        f"output/diffusion-dataset-3/ngo/precondition/powtheta-{pow}/{dataname}/A-DEF1-{sm}.pt"
    )
    iterses2.append(torch.load(path, weights_only=False)["iters"] if path.exists() else None)
    # Load NGO result (approach 2)
    path = Path(
        f"output/diffusion-dataset-3/ngo/precondition/powtheta-{pow}-notngo/{dataname}/A-DEF1-{sm}.pt"
    )
    iterses3.append(torch.load(path, weights_only=False)["iters"])

iterses = [iterses0, iterses1, iterses2, iterses3]
means = np.array([[its.mean() if its is not None else None for its in iters] for iters in iterses])
fails = np.array(
    [[(its == 5001).sum() if its is not None else None for its in iters] for iters in iterses]
)

for i, pow in enumerate(pows):
    s = f"{pow} & {(ratio**pow).mean():.2g}"
    for m, f in zip(means[:, i], fails[:, i]):
        if m is None:
            s += " & -"
        elif f == 0:
            s += f" & {m:.1f}"
        else:
            s += f" & {m:.1f} ({f})"
    print(s)
