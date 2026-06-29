from .tables import makeplot
from .styles import colors, alphas, styles

import torch
import matplotlib.pyplot as plt

n = 80
dataset = "advdiff-low-peclet-9000-10000"
sm = "gs4"


p1 = "output/advdiff"
p2 = f"precondition/{dataset}/fgmres/n{n}"
files = [
    f"{p1}/less-train/ngo-N50/{p2}/A-DEF1-gs4.pt",
    f"{p1}/less-train/ngo-N100/{p2}/A-DEF1-gs4.pt",
    f"{p1}/less-train/ngo-N200/{p2}/A-DEF1-gs4.pt",
    f"{p1}/less-train/ngo-N400/{p2}/A-DEF1-gs4.pt",
    f"{p1}/less-train/ngo-N800/{p2}/A-DEF1-gs4.pt",
    f"{p1}/ngo/{p2}/A-DEF1-gs4.pt",
    f"output/projection/{p2}/A-DEF1-gs4.pt",
    # f"output/projection/{p2}/gs4.pt",
]
names = [
    "NGO, $N=50$",
    "NGO, $N=100$",
    "NGO, $N=200$",
    "NGO, $N=400$",
    "NGO, $N=800$",
    "NGO, $N=8000$",
    "Exact",
    # "Smoother only",
]

iterses = []
plt.figure(figsize=(5.0, 3.5))
for file in files:
    data = torch.load(file, weights_only=False)
    iterses.append(data["iters"])


plt.boxplot(iterses[::-1], orientation="horizontal", whis=(5, 95))
plt.yticks(range(1, 1 + len(files)), names[::-1])
plt.xlabel("F-GMRES iterations")
plt.tight_layout()
plt.savefig(f"plots/advdiff/datasetsize.pdf", transparent=True)
plt.close()
