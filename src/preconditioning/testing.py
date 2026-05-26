import math

import matplotlib.pyplot as plt

from src import *


def stack_all(arrays):
    n = len(arrays)
    M = max(array.shape[0] for array in arrays)
    out = np.full((M, n), np.NAN)
    for i, array in enumerate(arrays):
        k = len(array)
        out[:k, i] = array

    return out


def convergence_plot(logs, **kwargs):
    xlog = False
    if "xlog" in kwargs:
        xlog = kwargs["xlog"]
        del kwargs["xlog"]

    if "label" in kwargs:
        label = kwargs["label"]
        del kwargs["label"]
        kwargs1 = dict(kwargs, alpha=1)
        plt.plot([math.nan], **kwargs1, label=label)

    ns = range(1, 1 + logs.shape[0])
    plt.plot(ns, logs, **kwargs)
    plt.yscale("log")
    if xlog:
        plt.xscale("log")


def add_results(entries):
    tot_calls = 0
    tot_iters = 0
    for entry in entries:
        entry["info"] = torch.load("output/" + entry["path"] + ".pt", weights_only=False)
        entry["logs"] = torch.load("output/" + entry["path"] + ".logs.pt", weights_only=False)

    for entry in entries:
        its = entry["info"]["iters"]
        tot_calls += len(its)
        tot_iters += sum(its)
        failures = len([it for it in its if it == 1000])
        its = np.array([it for it in its if it != 1000])
        entry["failures"] = failures
        entry["converged-iters"] = its
        if len(its) == 0:
            entry["mean"] = np.nan
            entry["std"] = np.nan
            entry["min"] = np.nan
            entry["max"] = np.nan
            entry["median"] = np.nan
        else:
            entry["mean"] = np.mean(its)
            entry["std"] = np.std(its)
            entry["min"] = np.min(its)
            entry["max"] = np.max(its)
            entry["median"] = np.median(its)

    return tot_calls, tot_iters
