from timeit import default_timer as timer
import os
from pathlib import Path

from tqdm import trange
import torch
import numpy as np
import scipy
from scipy.sparse.linalg import aslinearoperator
import ilupp
import pyamg

from ..dataset import DataSet
from ..utils import save
from ..utils.quadrature import TensorGrid
from .linear_system import LinearSystem
from .assemble import assemble
from ..discretise import _discretise_batch, _make_attributes
from .linear_maps import Kronecker, Identity, Chain
from ..training.models.ngo import NGO
from .smoothers import BlockJacobiSmoother, DiffusionSmoother, DownwindGS, JacobiSmoother
from .coarse_solvers import make_Q


def _deflation_preconditioner(A, Q, M, kind):
    """
        def make_deflation_preconditioner(A, Q, M, kind)

    Construct a deflation method preconditioner.

    - A: the original matrix in the linear system
    - Q: the coarse solver
    - M: the "classical" preconditioner, i.e. incomplete Cholesky, Schwarz, ...
    - kind: which deflation method to use
    """

    if kind == "None" or kind is None:
        return scipy.sparse.linalg._interface.IdentityOperator((n, n), P.dtype)
    elif kind == "PREC":
        return M
    elif kind == "AD":
        return M + Q
    elif kind == "A-DEF1":
        return Chain(A, (Q, M))
    elif kind == "A-DEF2":
        return Chain(A, (M, Q))
    elif kind == "BNN":
        return Chain(A, (Q, M, Q))
    elif kind == "BNN-b":
        A = aslinearoperator(A)
        n = A.shape[0]
        I = scipy.sparse.linalg._interface.IdentityOperator((n, n), A.dtype)
        Pr = I - A @ Q
        return Pr.T @ M @ Pr + Q
    else:
        raise ValueError(f'unknown preconditioner kind "{kind}"')


def make_smoother(batch_d, A, nodes, ids, config):
    nodes_x, nodes_y = nodes
    ix, iy = ids
    M_kind = config.get("M", None)
    if M_kind is None:
        return Identity(A.shape[0], np.float64)
    elif M_kind == "jacobi":
        return (4 / 5) * JacobiSmoother(A)
    elif M_kind.startswith("blockjacobi_"):
        # Format: blockjacobi_Bx_By, where Bx and By are the numbers of blocks in the x- and y-directions
        params = M_kind.split("_")[1:]
        Bx = int(params[0])
        By = int(params[1])
        return (4 / 5) * BlockJacobiSmoother(A, len(ix), len(iy), Bx, By)
    elif M_kind == "ilu":
        return ilupp.ILU0Preconditioner(A)
    elif M_kind == "ichol":
        return ilupp.IChol0Preconditioner(A)
    elif M_kind == "amg":
        return pyamg.ruge_stuben_solver(A).aspreconditioner()
    elif M_kind.startswith("G_diffusion"):
        # Smoother format: "G_diffusion_H_wmin_r"
        # For example: G_diffusion_0.05_2_1.5
        parts = M_kind.split("_")[2:]
        H = float(parts[0])
        wmin = int(parts[1])
        r = float(parts[2])
        h = (nodes_x[1] - nodes_x[0]).item()
        pow = config.get("powtheta", 1)
        theta = np.array(batch_d["theta"][0, ix][:, iy] ** pow)
        return DiffusionSmoother(h, max(wmin, int(H / h)), theta, r)
    elif M_kind.startswith("G_helmholtz_"):
        parts = M_kind.split("_")[2:]
        H = float(parts[0])
        wmin = int(parts[1])
        r = float(parts[2])
        h = (nodes_x[1] - nodes_x[0]).item()
        k = np.array(batch_d["theta"][0, ix][:, iy])
        return DiffusionSmoother(h, max(wmin, int(H / h)), np.ones_like(k), r)
    elif M_kind == "gs4":
        xs = np.array(nodes_x[ix])
        ys = np.array(nodes_y[iy])
        P1 = DownwindGS(xs, ys, A, 1.0, 1.0)
        P2 = DownwindGS(xs, ys, A, -1.0, 1.0)
        P3 = DownwindGS(xs, ys, A, -1.0, -1.0)
        P4 = DownwindGS(xs, ys, A, 1.0, -1.0)
        return Chain(A, (P1, P2, P3, P4))
    else:
        raise ValueError(f"invalid preconditioner string {M_kind}")


def _assert_prop_in(prefix, value, allowed):
    if value not in allowed:
        raise ValueError(f'{prefix} has illegal value "{value}" (allowed: {allowed})')


def make_preconditioned_system(A, b, batch, model, config, quad, ids):
    solver = config.get("solver")
    if model is None:
        return LinearSystem(A, b, solver=solver, kwargs=config.get("solver_kwargs", {}))

    ix, iy = ids
    with_x0 = config.get("with_x0", False)
    side = config.get("side", None)
    pe_max = config.get("pe_max", None)
    _assert_prop_in(
        "solver", solver, ("gmres", "fgmres", "bicgstab", "cg", "gmresinf", "fgmresinf")
    )
    _assert_prop_in("with_x0", with_x0, (True, False))
    _assert_prop_in("side", side, (None, "left", "right", "inner"))
    # Use the model to obtain the matrix
    # Combine this with the basis to obtain a preconditioner
    nodes_x, weights_x, nodes_y, weights_y = quad

    prob = batch["problem"]
    del batch["problem"]
    attributes = _make_attributes(
        batch, (nodes_x, weights_x), (nodes_y, weights_y), device="cpu", dtype=torch.float32
    )
    batch["problem"] = prob
    batch_d = _discretise_batch(batch, attributes, torch.float32)
    batch_d["problem"] = prob
    batch_d.update(attributes)

    t0 = timer()
    Q = make_Q(model, A, batch, quad, ids, config)
    t_prec = timer() - t0
    M = make_smoother(batch_d, A, (nodes_x, nodes_y), ids, config)
    P = _deflation_preconditioner(A, Q, M, config["kind"])

    system_kwargs = {"solver": solver, "kwargs": config["solver_kwargs"]}
    if with_x0:
        system_kwargs["x0"] = values.ravel()
    if side is not None:
        if side == "left":
            system_kwargs["P_left"] = P
        elif side == "right":
            system_kwargs["P_right"] = P
        else:
            system_kwargs["P_inner"] = P

    return LinearSystem(A, b, **system_kwargs, t_prec=t_prec)


def test_preconditioning(config, desc=None, save_results=True, print_summary=True):
    # 1. For each problem in the dataset
    #    Assemble the linear system
    #    Construct the preconditioner
    #    Solve the system
    dataset_name = config["dataset_name"]
    model_path = config["model_path"]
    dataset = DataSet.load(dataset_name)
    if model_path is not None:
        if not model_path.endswith(".pt"):
            model_path += "/model.pt"
        model = torch.load(f"output/{model_path}", weights_only=False)
        # Remove file from the path, so that preconditioning results are in a subfolder
        idx = model_path.rfind("/")
        model_path = model_path[:idx]
    else:
        model = None

    N = len(dataset)
    nx, ny = config["grid_size"]
    iter = range(N) if desc is None else trange(N, desc=desc, ascii=True, dynamic_ncols=True)
    kwargs = {}
    ts_assemble = np.zeros(N)
    ts_prec = np.zeros(N)
    ts_solve = np.zeros(N)
    relerrs = np.zeros(N)
    iters = np.zeros(N, dtype=np.int32)
    log = []
    random_rhs = config.get("random-rhs", False)
    seed = config.get("seed", None)
    if seed is not None:
        np.random.seed(seed)

    for i in iter:
        batch = dataset.get_batch(range(i, i + 1), "cpu")

        # Assemble linear system
        t0 = timer()
        A, b, quad, (ix, iy) = assemble(batch, nx, ny, config)
        t_assemble = timer() - t0
        if random_rhs:
            b = np.random.randn(b.size)

        normb = np.linalg.norm(b)
        system = make_preconditioned_system(A, b, batch, model, config, quad, (ix, iy))
        system.kwargs["rtol"] = 0.0
        system.kwargs["atol"] = config["tol"] * normb
        system.solve()
        r = A @ system.sol - b
        relerr = np.linalg.norm(r) / normb
        s = f"{len(system.log):>5} its, {system.t:>5.2f}s, {relerr =:>10.3g}"
        if desc is None:
            print(s)
        else:
            iter.set_description_str(desc + " " + s)

        ts_assemble[i] = t_assemble
        ts_prec[i] = system.t_prec
        ts_solve[i] = system.t
        relerrs[i] = relerr
        iters[i] = len(system.log)
        log.append(system.log)

    if print_summary:
        print(f"Mean iters: {iters.mean()}, median iters: {np.median(iters)}")

    if not save_results:
        return

    output = {
        "dataset": dataset_name,
        "model": model_path,
        "config": config,
        "ts_assemble": ts_assemble,
        "ts_prec": ts_prec,
        "ts_solve": ts_solve,
        "relerrs": relerrs,
        "iters": iters,
        "logs": log,
    }
    if model_path is not None:
        path = Path(f"output/{model_path}/precondition")
    else:
        path = Path(f"output/no-model/precondition")

    if not path.exists():
        os.makedirs(path)

    filename = config.get("filename")
    save(output, path / (filename + ".pt"), allow_exists=True)
