# 3D equivalents of the functions in `src/preconditioning/coarse_solvers.py` and
# `src/preconditioning/test_preconditioning.py`. Preconditioning tests for 3D data sets are run
# with the same `test_preconditioning` function as in 2D, which dispatches to the functions here.
from timeit import default_timer as timer

import ilupp
import numpy as np
import pyamg
import torch
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from src.preconditioning.linear_maps import Kronecker, Identity
from src.preconditioning.linear_system import LinearSystem
from src.preconditioning.smoothers import JacobiSmoother
from src.preconditioning.test_preconditioning import _assert_prop_in, _deflation_preconditioner
from src.training.models import ProjectOnBasis
from .assemble import assemble_diffusion_3d
from .basis import tensor_contract
from .discretise import _make_attributes_3d, _discretise_batch_3d
from .ngo2 import NGO_3D


def _discretise_for_model(batch, quad, dtype):
    # Discretise a batch of functions of size 1 on the grid of the finite volume discretisation
    batch = dict(batch)
    problem = batch.pop("problem")
    nodes_x, weights_x, nodes_y, weights_y, nodes_z, weights_z = quad
    attributes = _make_attributes_3d(
        batch,
        (nodes_x, weights_x),
        (nodes_y, weights_y),
        (nodes_z, weights_z),
        device="cpu",
        dtype=dtype,
    )
    batch_d = _discretise_batch_3d(batch, attributes, dtype)
    batch_d["problem"] = problem
    batch_d.update(attributes)
    return batch_d


def _remove_dofs(matrix, shape, noleftright):
    # Remove the basis functions on the Dirichlet boundaries from the (dense) coarse matrix
    if noleftright:
        to_remove = np.full(shape, False)
        to_remove[0, :, :] = True
        to_remove[-1, :, :] = True
        to_remove = to_remove.ravel()
        matrix[:, to_remove] = 0
        matrix[to_remove, :] = 0


def make_Q_3d(model, A, batch, quad, ids, config):
    """
    3D equivalent of `make_Q`.

    - `model`: the model to construct the coarse solver from. Either a NGO_3D, or a ProjectOnBasis
      with a TensorBasis3D, in which case the preconditioner is based on exact deflation
    - `A`: the matrix to precondition
    - `batch`: a non-discretised batch of size 1 (i.e. the fields are still functions)
    - `quad`: tuple (nodes_x, weights_x, nodes_y, weights_y, nodes_z, weights_z)
    - `ids`: tuple (ix, iy, iz) of indices into the quadrature nodes and weights representing which
      points are present in the discretisation
    - `config`: other options. Supported are "symmetrise", "noleftright", and "powtheta"
    """
    if isinstance(model, NGO_3D):
        if config.get("symmetrise", False):
            return make_Q_ngo_3d(model, A, batch, quad, ids, config)
        else:
            return ContinuousModelWrapper3D(model, batch, quad, ids, config)
    elif isinstance(model, ProjectOnBasis):
        return make_Q_projection_3d(model, A, batch, quad, ids, config)
    else:
        raise TypeError(f"No implementation for 3D coarse solver from {type(model)}")


def _restriction(basis, quad, ids):
    # Restriction operator: integrating against the basis
    nodes = quad[0::2]
    factors = [np.array(f(n), dtype=np.float64)[:, i] for (f, n, i) in zip(basis, nodes, ids)]
    return Kronecker(*factors), tuple(F.shape[0] for F in factors)


def make_Q_projection_3d(model, A, batch, quad, ids, config):
    basis = (model.basis.f_x, model.basis.f_y, model.basis.f_z)
    R, shape = _restriction(basis, quad, ids)
    # Prolongation operator: expanding in basis
    P = R.T
    E = R @ aslinearoperator(A) @ P
    if config.get("symmetrise", False):
        E = (E + E.T) / 2

    E = E @ np.eye(E.shape[0])
    _remove_dofs(E, shape, config.get("noleftright", False))
    Einv = aslinearoperator(np.linalg.pinv(E))
    return P @ Einv @ R


def make_Q_ngo_3d(model, A, batch, quad, ids, config):
    batch_d = _discretise_for_model(batch, quad, torch.float32)
    old_outputs = model.outputs
    model.outputs = ("matrices", "bases")
    model.train(False)
    with torch.no_grad():
        outputs = model(batch_d)
    model.outputs = old_outputs

    basis = (model.basis.f_x, model.basis.f_y, model.basis.f_z)
    R, shape = _restriction(basis, quad, ids)
    # Prolongation operator: expanding in basis
    P = R.T

    # Convert from FEM to FDM scaling
    matrix = np.array(outputs["matrices"][0], dtype=np.float64)
    scale = 1.0
    for weights in quad[1::2]:
        scale *= (np.sum(np.array(weights)) / len(weights)).item()
    if config.get("symmetrise", False):
        matrix = (matrix + matrix.T) * 0.5
    matrix = matrix * scale

    _remove_dofs(matrix, shape, config.get("noleftright", False))
    return P @ aslinearoperator(matrix) @ R


class ContinuousModelWrapper3D(LinearOperator):
    def __init__(self, model, batch, quad, ids, config):
        """
        3D equivalent of `ContinuousModelWrapper`: a linear operator that applies the model to a
        right-hand side `f` sampled on the grid points of the discretisation (with zero boundary
        data), and returns the resulting solution on the same grid points.
        """
        self.model = model.to_double()
        self.model.train(False)
        self.model.outputs = ("coeffs", "bases")

        batch_d = _discretise_for_model(batch, quad, torch.float64)
        if (pow := config.get("powtheta_ngo")) is not None:
            batch_d["theta"] = batch_d["theta"] ** pow
        elif (pow := config.get("powtheta")) is not None:
            # As in 2D: don't take powers of theta as input to the NGO, but rescale so that the
            # preconditioner has the right magnitude
            weights = (batch_d["weights_x"], batch_d["weights_y"], batch_d["weights_z"])
            th_avg = torch.einsum("Bxyz,x,y,z->B", batch_d["theta"], *weights)
            th_avg_pow = torch.einsum("Bxyz,x,y,z->B", batch_d["theta"] ** pow, *weights)
            batch_d["theta"] = batch_d["theta"] * th_avg_pow / th_avg

        self.batch = batch_d
        self.batch["f"] = torch.zeros_like(self.batch["f"])
        for key in ("dirichlet:g", "neumann_y:eta", "neumann_z:eta"):
            self.batch[key] = torch.zeros_like(self.batch[key])

        self.ids = ids
        self.grid_shape = tuple(len(i) for i in ids)
        self.dtype = np.float64
        N = int(np.prod(self.grid_shape))
        self.shape = (N, N)
        self.noleftright = config.get("noleftright", False)

    def _matvec(self, x):
        f = np.reshape(x, self.grid_shape)
        ix, iy, iz = self.ids
        # See ContinuousModelWrapper for why slices are used here instead of the index arrays
        self.batch["f"][0, ix[0] : ix[-1] + 1, iy[0] : iy[-1] + 1, iz[0] : iz[-1] + 1] = (
            torch.tensor(f)
        )

        with torch.no_grad():
            output = self.model(self.batch)

        coeffs = output["coeffs"][0]
        if self.noleftright:
            coeffs[0, :, :] = 0
            coeffs[-1, :, :] = 0

        values = tensor_contract(
            coeffs,
            output["basis_x"][:, ix],
            output["basis_y"][:, iy],
            output["basis_z"][:, iz],
        )
        return np.array(values.flatten())


def make_smoother_3d(A, config):
    """
    Construct the "classical" preconditioner M. Only the smoothers that do not depend on the
    geometry of the problem are supported in 3D.
    """
    M_kind = config.get("M", None)
    if M_kind is None:
        return Identity(A.shape[0], np.float64)
    elif M_kind == "jacobi":
        return JacobiSmoother(A)
    elif M_kind == "ilu":
        return ilupp.ILU0Preconditioner(A)
    elif M_kind == "ichol":
        return ilupp.IChol0Preconditioner(A)
    elif M_kind == "amg":
        return pyamg.ruge_stuben_solver(A).aspreconditioner()
    else:
        raise ValueError(f"invalid or unsupported 3D preconditioner string {M_kind}")


def make_preconditioned_system_3d(A, b, batch, model, config, quad, ids):
    """
    3D equivalent of `make_preconditioned_system`.
    """
    solver = config.get("solver")
    if model is None:
        return LinearSystem(A, b, solver=solver, kwargs=config.get("solver_kwargs", {}))

    side = config.get("side", None)
    _assert_prop_in(
        "solver", solver, ("gmres", "fgmres", "bicgstab", "cg", "gmresinf", "fgmresinf")
    )
    # Using the model output as initial guess is not supported (yet) in 3D
    _assert_prop_in("with_x0", config.get("with_x0", False), (False,))
    _assert_prop_in("side", side, (None, "left", "right", "inner"))

    t0 = timer()
    Q = make_Q_3d(model, A, batch, quad, ids, config)
    t_prec = timer() - t0
    M = make_smoother_3d(A, config)
    P = _deflation_preconditioner(A, Q, M, config["kind"])

    system_kwargs = {"solver": solver, "kwargs": config["solver_kwargs"]}
    if side == "left":
        system_kwargs["P_left"] = P
    elif side == "right":
        system_kwargs["P_right"] = P
    elif side == "inner":
        system_kwargs["P_inner"] = P

    return LinearSystem(A, b, **system_kwargs, t_prec=t_prec)
