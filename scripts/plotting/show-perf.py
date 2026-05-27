import torch
import matplotlib.pyplot as plt
import numpy as np
from .styles import *

ns = [20, 40, 80, 160, 320]

timings = np.zeros((len(ns), 3))

for i, n in enumerate(ns):
    infile = f"scripts/plotting/cache/timings-n{n}-desktop-1t.pt"
    outfile = f"plots/diffusion/timings-n{n}.pdf"
    alldata = torch.load(infile, weights_only=False)

    ts_assemble = alldata["ts_assemble"]
    ts_construct_M = alldata["ts_construct_M"]
    ts_construct_Q_exact = alldata["ts_construct_Q_exact"]
    ts_construct_Q_ngo = alldata["ts_construct_Q_ngo"]
    ts_construct_P_AMG = alldata["ts_construct_P_AMG"]
    its = alldata["its"]
    ts_solve = alldata["ts_solve"]
    ts_eval_M = alldata["ts_eval_M"]
    ts_eval_Q_exact = alldata["ts_eval_Q_exact"]
    ts_eval_Q_ngo = alldata["ts_eval_Q_ngo"]
    ts_eval_P_AMG = alldata["ts_eval_P_AMG"]

    # Bars:
    # 1. Construct Q
    # 2. Construct M
    # 3. Eval Q
    # 4. Eval M
    # 5. CG

    # Question: should be use mean() or min()? Especially for the stuff that calls into torch, the
    # difference can be huge. I'm in favour of using min() for constructing the preconditioners, since
    # this is the same amount of numerical work each time.

    tcq = np.array(
        [ts_construct_Q_exact.mean(), ts_construct_Q_ngo.mean(), ts_construct_P_AMG.mean()]
    )
    tcm = np.array([ts_construct_M.mean(), ts_construct_M.mean(), 0.0])
    teq = np.array([ts_eval_Q_exact.mean(), ts_eval_Q_ngo.mean(), ts_eval_P_AMG.mean()])
    tem = np.array([ts_eval_M[0].mean(), ts_eval_M[1].mean(), 0.0])
    ts = ts_solve.mean(axis=1) - teq - tem

    bottom = np.zeros(3)
    entries = [
        ("Construct Q", tcq),
        ("Construct M", tcm),
        ("Evaluate Q", teq),
        ("Evaluate M", tem),
        ("Other", ts),
    ]
    names = ["Exact two-level", "NGO two-level", "AMG"]

    # Print a table
    # Columns: mean CG iters, setup Q, setup M, solve Q, solve M, total
    tab = np.zeros((3, 6))
    tab[:, 0] = np.mean(its, axis=1)

    tab[0, 1] = np.mean(ts_construct_Q_exact)
    tab[1, 1] = np.mean(ts_construct_Q_ngo)
    tab[2, 1] = np.mean(ts_construct_P_AMG)

    tab[0, 2] = np.mean(ts_construct_M)
    tab[1, 2] = np.mean(ts_construct_M)

    tab[0, 3] = np.mean(ts_eval_Q_exact)
    tab[1, 3] = np.mean(ts_eval_Q_ngo)
    tab[2, 3] = np.mean(ts_eval_P_AMG)

    tab[0, 4] = np.mean(ts_eval_M[0])
    tab[1, 4] = np.mean(ts_eval_M[1])

    tab[:, 5] = np.mean(ts_solve, axis=1) + np.sum(tab[:, 1:3], axis=1)
    timings[i] = tab[:, 5]

    # Convert times to milliseconds
    tab[:, 1:] *= 1e3
    tab[tab == 0] = np.nan
    print(infile)
    for row in tab:
        print("   ".join(f"{x:>6.1f}" for x in row))

    print()

    plt.figure(figsize=(5.0, 3.5))
    for name, times in entries:
        plt.bar(names, times, label=name, bottom=bottom)
        bottom += times

    plt.ylabel("Time taken")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outfile, transparent=True)
    plt.close()

plt.figure(figsize=(5.0, 3.5))
plt.plot(timings, label=["Exact two-level", "NGO two-level", "AMG"], linestyle="--", marker="o")
plt.legend()
plt.yscale("log")
plt.ylabel("Solve time (s)")
plt.xticks(range(len(ns)), [f"$n = {n}$" for n in ns])
plt.tight_layout()
plt.savefig("plots/diffusion/timings.pdf", transparent=True)
plt.close()
