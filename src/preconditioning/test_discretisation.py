# Script to test the accuracy of the finite volume / finite difference discretisations
from tqdm import trange
import scipy
import numpy as np
import matplotlib.pyplot as plt

from ..dataset import DataSet
from ..utils.quadrature import TensorGrid
from .assemble import assemble


def test_discretisation(dataset, nx, ny):
    dataset = DataSet.load(dataset)
    N = len(dataset)
    iter = trange(N, desc="Testing discretisation")
    kwargs = {}
    for i in iter:
        batch = dataset.get_batch(range(i, i + 1), "cpu")
        # Assemble linear system. Use direct solver for now
        A, b, quad = assemble(batch, nx, ny)
        u_pred = scipy.sparse.linalg.spsolve(A, b)
        u_pred = np.reshape(u_pred, (nx - 2, ny))

        # Also evaluate reference solution on the relevant grid
        nodes_x, weights_x, nodes_y, weights_y = quad
        grid = TensorGrid(nodes_x, nodes_y)
        u_fem = np.array(batch["u"][0](grid).reshape(nx, ny)[1:-1, :])

        err = u_pred - u_fem
        err_norm = np.sqrt(np.einsum("xy,xy,x,y", err, err, weights_x[1:-1], weights_y))
        fem_norm = np.sqrt(np.einsum("xy,xy,x,y", u_fem, u_fem, weights_x[1:-1], weights_y))
        iter.write(f"relerr: {err_norm/fem_norm}")


if __name__ == "__main__":
    test_discretisation("test", 101, 100)
