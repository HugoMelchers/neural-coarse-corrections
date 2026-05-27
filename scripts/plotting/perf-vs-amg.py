# Compare performance of preconditioners on diffusion:
# 1. Exact two-level
# 2. NGO-based two-level
# 3. Algebraic multigrid with PyAMG
# Also obtain a breakdown of time spent on
# - Assembling the linear system
# - Constructing preconditioners
# - Solving
#   - Calling the smoother / coarse solver / AMG preconditioner
import torch
import numpy as np
import pyamg
from scipy.sparse.linalg import cg, aslinearoperator
from tqdm import trange
from timeit import default_timer as timer

from src import DataSet, TensorGrid, NGO
from src.preconditioning.test_preconditioning import make_smoother, Chain, assemble, Kronecker
from src.discretise import _discretise_batch, _make_attributes
from logging_linop import Timed

dataset_name = "diffusion-dataset-3:9000-10000"
n = 80
tol = 4e-6
outfile = f"scripts/plotting/cache/timings-n{n}-desktop-1t"
projection_path = "projection"
ngo_path = "diffusion/ngo"
smoother = "G_diffusion_0.05_1_1.5"

nx, ny = (n + 1, n)

projection = torch.load(f"output/{projection_path}/model.pt", weights_only=False)
ngo = torch.load(f"output/{ngo_path}/model.pt", weights_only=False)
dataset = DataSet.load(dataset_name)
N = len(dataset)
iter = trange(N, ascii=True, dynamic_ncols=True)


def make_Q_projection(model, A, nodes, ids, config, cache):
    nodes_x, nodes_y = nodes
    # basis_x = np.array(model.basis.f_x(nodes_x), dtype=np.float64)
    # basis_y = np.array(model.basis.f_y(nodes_y), dtype=np.float64)
    basis_x = cache["basis_x"]
    basis_y = cache["basis_y"]
    ix, iy = ids

    symmetrise = config.get("symmetrise", False)
    noleftright = config.get("noleftright", False)
    nobottomtop = config.get("nobottomtop", False)

    # Restriction operator: integrating against basis
    R = Kronecker(basis_x[:, ix], basis_y[:, iy])
    # Prolongation operator: expanding in basis
    P = R.T
    E = R @ aslinearoperator(A) @ P
    if symmetrise:
        E = (E + E.T) / 2

    def remove_dofs(arr):
        if noleftright or nobottomtop:
            to_remove = np.full((12, 12), False)
            if noleftright:
                to_remove[0, :] = True
                to_remove[-1, :] = True
            if nobottomtop:
                to_remove[:, 0] = True
                to_remove[:, -1] = True
            to_remove = to_remove.ravel()
            arr[:, to_remove] = 0
            arr[to_remove, :] = 0

    E = E @ np.eye(E.shape[0])
    remove_dofs(E)
    Einv = aslinearoperator(np.linalg.pinv(E))
    Q = P @ Einv @ R
    return Q


def make_Q_ngo(model, A, batch_d, quad, ids, config, cache):
    nodes_x, weights_x, nodes_y, weights_y = quad
    ix, iy = ids

    nodes = (nodes_x, nodes_y)
    nx, ny = len(nodes_x), len(nodes_y)
    grid = TensorGrid(nodes_x, weights_y)
    model.outputs = ("values", "matrices", "bases")

    model.train(False)
    with torch.no_grad():
        K, b, scale = model.assemble(batch_d)
        _, matrices = model.perform_matmul(K, b)
        matrices = matrices * scale[:, None, None]

    model.train(True)

    matrix = np.array(matrices[0])

    basis_x = cache["basis_x"]
    basis_y = cache["basis_y"]
    basis = (basis_x, basis_y)

    weights_x = np.array(weights_x, dtype=np.float64)
    weights_y = np.array(weights_y, dtype=np.float64)
    weights = (weights_x, weights_y)

    symmetrise = config.get("symmetrise", False)
    noleftright = config.get("noleftright", False)
    nobottomtop = config.get("nobottomtop", False)

    basis_x, basis_y = basis
    nodes_x, nodes_y = nodes
    weights_x, weights_y = weights
    ix, iy = ids
    # Restriction operator: integrating against basis
    R = Kronecker(basis_x[:, ix], basis_y[:, iy])
    # Prolongation operator: expanding in basis
    P = R.T

    def remove_dofs(arr):
        if noleftright or nobottomtop:
            to_remove = np.full((12, 12), False)
            if noleftright:
                to_remove[0, :] = True
                to_remove[-1, :] = True
            if nobottomtop:
                to_remove[:, 0] = True
                to_remove[:, -1] = True
            to_remove = to_remove.ravel()
            arr[:, to_remove] = 0
            arr[to_remove, :] = 0

    # Convert to full matrix
    matrix = matrix @ np.eye(matrix.shape[0])
    # Convert from FEM to FDM scaling
    hx = np.sum(weights_x) / len(weights_x)
    hy = np.sum(weights_y) / len(weights_y)
    scale = hx * hy
    if symmetrise:
        Einv = aslinearoperator((matrix + matrix.T) * scale * 0.5)
    else:
        Einv = aslinearoperator(matrix * scale)

    remove_dofs(Einv.A)
    Q = P @ Einv @ R
    return Q


kwargs = {}
ts_assemble = np.zeros(N)
ts_construct_M = np.zeros(N)
ts_construct_Q_exact = np.zeros(N)
ts_construct_Q_ngo = np.zeros(N)
ts_construct_P_AMG = np.zeros(N)
its = np.zeros((3, N), dtype=np.int32)
ts_solve = np.zeros((3, N))
ts_eval_M = np.zeros((2, N))
ts_eval_Q_exact = np.zeros(N)
ts_eval_Q_ngo = np.zeros(N)
ts_eval_P_AMG = np.zeros(N)

for i in iter:
    batch = dataset.get_batch(range(i, i + 1), "cpu")

    # Assemble linear system
    t0 = timer()
    A, b, quad, ids = assemble(batch, nx, ny)
    ts_assemble[i] = timer() - t0
    (nodes_x, weights_x, nodes_y, weights_y) = quad
    ix, iy = ids

    prob = batch["problem"]
    del batch["problem"]
    nodes = (nodes_x, nodes_y)
    attributes = _make_attributes(
        batch, (nodes_x, weights_x), (nodes_y, weights_y), device="cpu", dtype=torch.float32
    )
    batch["problem"] = prob
    batch_d = _discretise_batch(batch, attributes, torch.float32)
    batch_d["problem"] = prob
    batch_d.update(attributes)
    basis_x = np.array(ngo.basis.f_x(nodes_x), dtype=np.float64)
    basis_y = np.array(ngo.basis.f_y(nodes_y), dtype=np.float64)
    cache = {"basis_x": basis_x, "basis_y": basis_y}

    # Construct smoother, coarse solvers, and AMG preconditioners
    make_smoother(batch_d, A, nodes, ids, {"M": smoother})
    t0 = timer()
    M = make_smoother(batch_d, A, nodes, ids, {"M": smoother})
    ts_construct_M[i] = timer() - t0

    make_Q_projection(projection, A, nodes, ids, {"noleftright": True}, cache)
    t0 = timer()
    Q_exact = make_Q_projection(projection, A, nodes, ids, {"noleftright": True}, cache)
    ts_construct_Q_exact[i] = timer() - t0

    make_Q_ngo(ngo, A, batch_d, quad, ids, {"noleftright": True, "symmetrise": True}, cache)
    t0 = timer()
    Q_ngo = make_Q_ngo(ngo, A, batch_d, quad, ids, {"noleftright": True, "symmetrise": True}, cache)
    ts_construct_Q_ngo[i] = timer() - t0

    pyamg.ruge_stuben_solver(A).aspreconditioner()
    t0 = timer()
    P_AMG = pyamg.ruge_stuben_solver(A).aspreconditioner()
    ts_construct_P_AMG[i] = timer() - t0

    # Construct A-DEF1 preconditioners (this is not timed since it only creates some wrappers)
    A_timed = Timed(aslinearoperator(A))
    P_exact = Chain(A, (Timed(Q_exact), Timed(M)))
    P_ngo = Chain(A, (Timed(Q_ngo), Timed(M)))
    P_AMG = Timed(P_AMG)

    # Solve! Each one with a warm-up
    sol, info = cg(A_timed, b, rtol=tol, M=P_exact)
    P_exact.Ps[0].ts = []
    P_exact.Ps[1].ts = []
    A_timed.ts = []
    t0 = timer()
    sol, info = cg(A_timed, b, rtol=tol, M=P_exact)
    ts_solve[0, i] = timer() - t0
    its[0, i] = len(A_timed.ts)
    A_timed.ts = []
    ts_eval_M[0, i] = sum(P_exact.Ps[1].ts)
    ts_eval_Q_exact[i] = sum(P_exact.Ps[0].ts)

    sol, info = cg(A_timed, b, rtol=tol, M=P_ngo)
    P_ngo.Ps[0].ts = []
    P_ngo.Ps[1].ts = []
    A_timed.ts = []
    t0 = timer()
    sol, info = cg(A_timed, b, rtol=tol, M=P_ngo)
    ts_solve[1, i] = timer() - t0
    its[1, i] = len(A_timed.ts)
    A_timed.ts = []
    ts_eval_M[1, i] = sum(P_ngo.Ps[1].ts)
    ts_eval_Q_ngo[i] = sum(P_ngo.Ps[0].ts)

    sol, info = cg(A_timed, b, rtol=tol, M=P_AMG)
    P_AMG.ts = []
    A_timed.ts = []
    t0 = timer()
    sol, info = cg(A_timed, b, rtol=tol, M=P_AMG)
    ts_solve[2, i] = timer() - t0
    its[2, i] = len(A_timed.ts)
    A_timed.ts = []
    ts_eval_P_AMG[i] = sum(P_AMG.ts)

    alldata = {
        "ts_assemble": ts_assemble,
        "ts_construct_M": ts_construct_M,
        "ts_construct_Q_exact": ts_construct_Q_exact,
        "ts_construct_Q_ngo": ts_construct_Q_ngo,
        "ts_construct_P_AMG": ts_construct_P_AMG,
        "its": its,
        "ts_solve": ts_solve,
        "ts_eval_M": ts_eval_M,
        "ts_eval_Q_exact": ts_eval_Q_exact,
        "ts_eval_Q_ngo": ts_eval_Q_ngo,
        "ts_eval_P_AMG": ts_eval_P_AMG,
    }
    torch.save(alldata, outfile + ".pt")
