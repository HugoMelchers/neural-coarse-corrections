import torch
import numpy as np

from src.discretise import _make_attributes, _discretise_batch
from src.utils.quadrature import TensorGrid
from .linear_maps import Kronecker
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from src.training.models import NGO, ProjectOnBasis, DeepONet, VarMiON, RINO
from src.utils.misc import einsum


def make_Q(model, A, batch, quad, ids, config):
    """
    - `model`: the machine learning model to construct the preconditioner from. Can also be a
      ProjectOnBasis, in which case the preconditioner is based on exact deflation
    - `A`: the matrix to precondition
    - `batch` a non-discretised batch of size 1 (i.e. the fields are still as functions)
    - `quad`: tuple (nodes_x, weights_x, modes_y, weights_y)
    - `ids`: tuple (xs, ys), slices into the quadrature nodes and weights representing which
    -   points are present in the discretisation
    - `config`: other options
    """
    if isinstance(model, NGO):
        if config.get("symmetrise", False):
            return make_Q_ngo(model, A, batch, quad, ids, config)
        else:
            return ContinuousModelWrapper(model, batch, quad, ids, config)
    elif isinstance(model, ProjectOnBasis):
        return make_Q_projection(model, A, batch, quad, ids, config)
    elif isinstance(model, DeepONet) or isinstance(model, VarMiON):
        return DiscreteModelWrapper(model, batch, quad, ids, config)
    elif isinstance(model, RINO):
        return ContinuousModelWrapper(model, batch, quad, ids, config)
    else:
        raise TypeError(f"No implementation for coarse solver from {type(model)}")


def make_Q_projection(model, A, batch, quad, ids, config):
    nodes_x, _, nodes_y, _ = quad
    basis_x = np.array(model.basis.f_x(nodes_x), dtype=np.float64)
    basis_y = np.array(model.basis.f_y(nodes_y), dtype=np.float64)
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


class DiscreteModelWrapper(LinearOperator):
    def __init__(self, model, batch, quad, ids, config):
        batch = dict(batch)
        self.model = model.to_double()
        self.model.train(False)
        old_outputs = model.outputs
        self.model.outputs = ("coeffs", "bases")
        nodes_x, weights_x, nodes_y, weights_y = quad
        quad_x = {"u": (nodes_x, weights_x), "default": "t12"}
        quad_y = {"u": (nodes_y, weights_y), "default": "t12"}
        ix, iy = ids

        prob = batch["problem"]
        del batch["problem"]
        attributes = _make_attributes(batch, quad_x, quad_y, device="cpu", dtype=torch.float64)
        batch_d = _discretise_batch(batch, attributes, torch.float64)
        batch_d["problem"] = prob
        batch_d.update(attributes)

        # If pe_max is set, approximate the Péclet number of this problem, and increase the
        # diffusion if the Péclet number exceeds the given maximum. (advection-diffusion only).
        pe_max = config.get("pe_max", None)
        if pe_max is not None:
            v_norm = batch_d["v_x"].hypot(batch_d["v_y"]).mean()
            theta = batch_d["theta"].mean()
            Pe = v_norm / theta
            if Pe > pe_max:
                ratio = Pe / pe_max
                batch_d["theta"] *= ratio

        self.batch = batch_d
        self.batch["dirichlet:g"] = torch.zeros_like(self.batch["dirichlet:g"])
        self.batch["neumann:eta"] = torch.zeros_like(self.batch["neumann:eta"])
        self.ids = ids
        self.nx = len(ix)
        self.ny = len(iy)
        self.dtype = np.float64
        N = self.nx * self.ny
        self.shape = (N, N)
        self.noleftright = config.get("noleftright", False)
        self.nobottomtop = config.get("nobottomtop", False)

        # Construct interpolation
        interp = config.get("interp", "nearest")
        if interp == "nearest":
            sensors = np.linspace(0, 1, 12)
            self.sample_x = (nodes_x[ix, None] - sensors[None, :]).abs().min(dim=0).indices
            self.sample_y = (nodes_y[iy, None] - sensors[None, :]).abs().min(dim=0).indices

    def _matvec(self, x):
        # Interpolate
        x = np.reshape(x, (self.nx, self.ny))
        f = x[self.sample_x, :][:, self.sample_y]
        # Write result in f, adding a singleton index because torch models work on batches
        self.batch["f"] = torch.tensor(f)[None]
        # Call model, remove batch index, and take selected point indices
        ix, iy = self.ids
        with torch.no_grad():
            output = self.model(self.batch)

        coeffs = output["coeffs"][0]
        basis_x, basis_y = output["basis_x"], output["basis_y"]
        if self.noleftright:
            coeffs[0, :] = 0
            coeffs[-1, :] = 0
        if self.nobottomtop:
            coeffs[:, 0] = 0
            coeffs[:, -1] = 0

        values = einsum("ix,ij,jy->xy", basis_x[:, ix], coeffs, basis_y[:, iy])
        # Give output
        return np.array(values.flatten())


class ContinuousModelWrapper(LinearOperator):
    def __init__(self, model, batch, quad, ids, config):
        batch = dict(batch)
        self.model = model.to_double()
        self.model.train(False)
        old_outputs = model.outputs
        self.model.outputs = ("coeffs", "bases")
        nodes_x, weights_x, nodes_y, weights_y = quad
        quad_x = (nodes_x, weights_x)
        quad_y = (nodes_y, weights_y)
        ix, iy = ids

        prob = batch["problem"]
        del batch["problem"]
        attributes = _make_attributes(batch, quad_x, quad_y, device="cpu", dtype=torch.float64)
        batch_d = _discretise_batch(batch, attributes, torch.float64)
        batch_d["problem"] = prob
        batch_d.update(attributes)

        # If pe_max is set, approximate the Péclet number of this problem, and increase the
        # diffusion if the Péclet number exceeds the given maximum. (advection-diffusion only).
        pe_max = config.get("pe_max", None)
        if pe_max is not None:
            v_norm = batch_d["v_x"].hypot(batch_d["v_y"]).mean()
            theta = batch_d["theta"].mean()
            Pe = v_norm / theta
            if Pe > pe_max:
                ratio = Pe / pe_max
                batch_d["theta"] *= ratio

        self.batch = batch_d
        self.batch["f"] = torch.zeros_like(self.batch["f"])
        self.batch["dirichlet:g"] = torch.zeros_like(self.batch["dirichlet:g"])
        self.batch["neumann:eta"] = torch.zeros_like(self.batch["neumann:eta"])
        self.ids = ids
        self.nx = len(ix)
        self.ny = len(iy)
        self.dtype = np.float64
        N = self.nx * self.ny
        self.shape = (N, N)
        self.noleftright = config.get("noleftright", False)
        self.nobottomtop = config.get("nobottomtop", False)

    def _matvec(self, x):
        f = np.reshape(x, (self.nx, self.ny))
        ix, iy = self.ids
        # The following line is very ugly, but has to be written in that specific way. What I *want*
        # to write is
        #
        #     self.batch["f"][0, ix, iy] = torch.tensor(f)
        #
        # However, since ix and iy are numpy arrays, this double indexing does not work the same way
        # as when ix and iy are slices (i.e. the objects that are implicitly used when indexing like
        # array[:, 3:7, 1:-1] etc). However, representing ix and iy as slices runs into other
        # interface-related issues. I could also write
        #
        #     self.batch["f"][0, ix, :][:, iy] = torch.tensor(f)
        #
        # This also doesn't work, though, as this indexing constructs a copy of self.batch["f"]
        # instead of a view into that tensor. Therefore the above line would do absolutely nothing
        # at all, as it would "update" the values of a temporary copy of batch["f"].
        #
        # To avoid these issues, I manually use slice indexing using the first and last entries of
        # the index arrays ix and iy. This means I can take the values of batch["f"] in one single
        # indexing expression, so that the assignment actually modifies the original tensor.
        self.batch["f"][0, ix[0] : ix[-1] + 1, iy[0] : iy[-1] + 1] = torch.tensor(f)
        # Call model, remove batch index, and take selected point indices
        with torch.no_grad():
            output = self.model(self.batch)

        coeffs = output["coeffs"][0]
        basis_x, basis_y = output["basis_x"], output["basis_y"]
        if self.noleftright:
            coeffs[0, :] = 0
            coeffs[-1, :] = 0
        if self.nobottomtop:
            coeffs[:, 0] = 0
            coeffs[:, -1] = 0

        values = einsum("ix,ij,jy->xy", basis_x[:, ix], coeffs, basis_y[:, iy])
        # Give output
        return np.array(values.flatten())


def make_Q_ngo(model, A, batch, quad, ids, config):
    batch = dict(batch)
    dtype = np.complex128 if batch["problem"] == "helmholtz" else np.float64
    old_outputs = model.outputs
    nodes_x, weights_x, nodes_y, weights_y = quad
    ix, iy = ids

    prob = batch["problem"]
    del batch["problem"]
    attributes = _make_attributes(
        batch, (nodes_x, weights_x), (nodes_y, weights_y), device="cpu", dtype=torch.float32
    )
    batch_d = _discretise_batch(batch, attributes, torch.float32)
    batch_d["problem"] = prob
    batch_d.update(attributes)

    nodes = (nodes_x, nodes_y)
    nx, ny = len(nodes_x), len(nodes_y)
    grid = TensorGrid(nodes_x, weights_y)
    model.outputs = ("values", "matrices", "bases")

    # If pe_max is set, approximate the Péclet number of this problem, and increase the
    # diffusion if the Péclet number exceeds the given maximum. (advection-diffusion only).
    pe_max = config.get("pe_max", None)
    if pe_max is not None:
        v_norm = batch_d["v_x"].hypot(batch_d["v_y"]).mean()
        theta = batch_d["theta"].mean()
        Pe = v_norm / theta
        if Pe > pe_max:
            ratio = Pe / pe_max
            batch_d["theta"] *= ratio

    model.train(False)
    with torch.no_grad():
        outputs = model(batch_d)

    model.train(True)

    values = np.array(outputs["values"][0, ix][:, iy], dtype=dtype)
    matrix = outputs["matrices"]
    matrix = np.array(matrix[0])

    basis_x = np.array(outputs["basis_x"], dtype=np.float64)
    basis_y = np.array(outputs["basis_y"], dtype=np.float64)
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
