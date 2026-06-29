import numpy as np
import pathlib
import matplotlib.pyplot as plt
from src.plotting import groupedbar
from src import *
from .styles import *


def fmt_table(table):
    nrow, ncol = table.shape
    tab_s = [[f"{entry:.1f}" for entry in row] for row in table]
    maxlen = max(len(s) for row in tab_s for s in row)
    tab_s = [[s.rjust(maxlen) for s in row] for row in tab_s]
    return "\n".join(" & ".join(row) for row in tab_s)


def makeplot(
    ns,
    *,
    solver,
    fileses,
    colours,
    alphas,
    seps,
    outfile,
    title=None,
    loc=None,
    outer=False,
    **kwargs,
):
    means1 = np.full((len(fileses), len(ns)), np.nan)
    means2 = np.full((len(fileses), len(ns)), np.nan)
    means1_conv = np.full((len(fileses), len(ns)), np.nan)
    means2_conv = np.full((len(fileses), len(ns)), np.nan)
    fails1 = np.full((len(fileses), len(ns)), -1)
    fails2 = np.full((len(fileses), len(ns)), -1)

    for i, (_, files) in enumerate(fileses):
        for j, file in enumerate(files):
            path = pathlib.Path(f"output/{file}.pt")
            if path.exists():
                results = torch.load(path, weights_only=False)
                maxiter = results["config"]["solver_kwargs"]["maxiter"]
                iters = results["iters"] - 1
                n_fails = (iters == maxiter).sum()
                fails1[i, j] = n_fails
                means1_conv[i, j] = iters[iters != maxiter].mean()
                if n_fails > 500:
                    continue
                means1[i, j] = iters.mean()
            else:
                print(f"{path} doesn't exist")

    for i, (_, files) in enumerate(fileses):
        for j, file in enumerate(files):
            path = pathlib.Path(f"output/{file}-lowtol.pt")
            if path.exists():
                results = torch.load(path, weights_only=False)
                maxiter = results["config"]["solver_kwargs"]["maxiter"]
                iters = results["iters"] - 1
                n_fails = (iters == maxiter).sum()
                fails2[i, j] = n_fails
                means2_conv[i, j] = iters[iters != maxiter].mean()
                if n_fails > 500:
                    continue
                means2[i, j] = iters.mean()

    print(outfile)
    print("Mean iterations (incl. failures)")
    print(fmt_table(means1))
    print()
    print("Mean iterations (excl. failures)")
    print(fmt_table(means1_conv))
    print()
    print("Failures")
    print(fmt_table(fails1))
    print()
    print()

    # If we omit the regular bar, also omit the overline
    means2[np.isnan(means1)] = np.nan

    groupedbar(
        means1,
        upper=means2,
        seps=seps,
        color=colours,
        alpha=alphas,
        label=[name for (name, _) in fileses],
        annotate=fails1 != 0,
    )
    plt.xticks(np.arange(len(ns)), [f"$n={n}$" for n in ns])
    plt.yscale("log")
    plt.ylabel(f"Mean {solver} iterations")
    plt.ylim(1, 1e6)
    plt.axhline(5000, c="grey", linestyle="--", linewidth=1.0)
    if loc is not None:
        if outer:
            ax = plt.gca()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
            plt.legend(loc=loc, bbox_to_anchor=(1.0, 0.5))
        else:
            plt.legend(loc=loc, **kwargs.get("lgnd", {"ncols": 2}))

    if title is None:
        title = f"{solver} preconditioner effectiveness"

    if title != "":
        plt.title(title)

    plt.tight_layout()
    plt.savefig(f"plots/{outfile}.pdf", transparent=True)
    plt.close()
