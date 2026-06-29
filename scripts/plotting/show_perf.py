import torch
import matplotlib.pyplot as plt

import numpy as np

from .styles import *

ns = [20, 40, 80, 160, 320]
ms = [6, 12, 24]

nn = len(ns)
nm = len(ms)

# We have AMG, and then for each m we have exact and NGO preconditioning
nprecs = 1 + 2 * nm
timings = np.zeros((len(ns), 3))
# Make Q, make M, eval Q, eval M, other_solve, total
nsteps = 6

timingdata = np.zeros((nn, nprecs, nsteps, 1000))
iterdata = np.zeros((nn, nprecs, 1000))

for idn, n in enumerate(ns):
    # Collect stats
    for idm, m in enumerate(ms):
        infile = f"scripts/plotting/cache/timings-n{n}-desktop-1t"
        if m != 12:
            infile += f"-{m}x{m}"
        infile += ".pt"

        # Load data
        alldata = torch.load(infile, weights_only=False)
        iterdata[idn, 1 + 2 * idm, :] = alldata["its"][0]
        iterdata[idn, 2 + 2 * idm, :] = alldata["its"][1]
        # Skip alldata["ts_assemble"]
        # Time to construct Q / M
        timingdata[idn, 1 + 2 * idm, 0, :] = alldata["ts_construct_Q_exact"]
        timingdata[idn, 2 + 2 * idm, 0, :] = alldata["ts_construct_Q_ngo"]
        timingdata[idn, 1 + 2 * idm, 1, :] = alldata["ts_construct_M"]
        timingdata[idn, 2 + 2 * idm, 1, :] = alldata["ts_construct_M"]
        # Time to evaluate Q / M
        timingdata[idn, 1 + 2 * idm, 2, :] = alldata["ts_eval_Q_exact"]
        timingdata[idn, 2 + 2 * idm, 2, :] = alldata["ts_eval_Q_ngo"]
        timingdata[idn, 1 + 2 * idm, 3, :] = alldata["ts_eval_M"][0]
        timingdata[idn, 2 + 2 * idm, 3, :] = alldata["ts_eval_M"][1]
        # Total time to solve
        timingdata[idn, 1 + 2 * idm, 4, :] = alldata["ts_solve"][0]
        timingdata[idn, 2 + 2 * idm, 4, :] = alldata["ts_solve"][1]

        # Also do AMG (once)
        if idm == 1:
            iterdata[idn, 0, :] = alldata["its"][2]
            timingdata[idn, 0, 0, :] = alldata["ts_construct_P_AMG"]
            timingdata[idn, 0, 2, :] = alldata["ts_eval_P_AMG"]
            timingdata[idn, 0, 4, :] = alldata["ts_solve"][2]

# collect the total times taken
# Note: this is make Q + make M + solve, since solve still includes evaluating Q and M
timingdata[:, :, -1] = timingdata[:, :, 0] + timingdata[:, :, 1] + timingdata[:, :, 4]
# from the solve times, subtract the times spent evaluating Q and M
timingdata[:, :, 4] -= timingdata[:, :, 2] + timingdata[:, :, 3]

timingmeans = timingdata.mean(axis=-1)
itermeans = iterdata.mean(axis=-1)

# Now plot:
# - AMG vs. exact and NGO two-level as function of n
colors = ["black", "C0", "C0", "C1", "C1", "C2", "C2"]
lines = ["-", "-", "--", "-", "--", "-", "--"]
markers = ["o", "o", "^", "o", "^", "o", "^"]
labels = [
    "AMG",
    "Exact $6\\times6$",
    "NGO $6\\times6$",
    "Exact $12\\times12$",
    "NGO $12\\times12$",
    "Exact $24\\times24$",
    "NGO $24\\times24$",
]
plt.figure(figsize=(5, 3.5))
for i in range(7):
    plt.plot(
        range(nn),
        timingmeans[:, i, -1],
        color=colors[i],
        linestyle=lines[i],
        label=labels[i],
        marker=markers[i],
    )

plt.xticks(range(nn), [f"$n={n}$" for n in ns])
plt.yscale("log")
plt.ylabel("Solve time [$s$]")
plt.ylim(1e-3 / 1.5, 150)
plt.tight_layout()
plt.legend(loc="upper left", ncols=2)
plt.savefig("plots/diffusion/timings.pdf", transparent=True)
plt.close()

# Print the bulk of the table (n=80)
for i in range(7):
    print(
        f"{labels[i]:<22}",
        f"{itermeans[2, i]:>4.1f}",
        f"{1e3 * timingmeans[2, i, 0]:>5.1f}",
        f"{1e3 * timingmeans[2, i, 1]:>4.1f}",
        f"{1e3 * timingmeans[2, i, 2]:>4.1f}",
        f"{1e3 * timingmeans[2, i, 3]:>4.1f}",
        f"{1e3 * timingmeans[2, i, 5]:>5.1f} \\\\",
        sep=" & ",
    )

for idn, n in enumerate(ns):
    # Stacked bar chart of the time breakdown for the n×n mesh
    plt.figure(figsize=(5, 3.5))
    ts_so_far = np.zeros(7)
    stepnames = [
        "Construct $\\mathbf{Q}$",
        "Construct $\\mathbf{M}$",
        "Evaluate $\\mathbf{Q}$",
        "Evaluate $\\mathbf{M}$",
        "Other",
    ]
    for step in range(5):
        ts = timingmeans[idn, :, step]
        plt.bar(range(7), ts, bottom=ts_so_far, label=stepnames[step])
        ts_so_far += ts

    # The exact 24x24 is always really slow, so don't bother fitting that in the plot. Instead, set
    # the y limits so that the second-slowest is nicely in view.
    ymax = np.sort(ts_so_far)[-2] * 1.1
    # plt.gca().tick_params("x", rotation=45, ha="right")  # , rotation_mode="xtick")
    plt.xticks(np.linspace(0, 6, 7), labels, ha="right", rotation=45)
    plt.ylim(0.0, ymax)
    plt.ylabel("Time taken [$s$]")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"plots/diffusion/timings-n{n}.pdf", transparent=True)
    plt.close()
