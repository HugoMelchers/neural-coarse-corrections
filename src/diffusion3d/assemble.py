import numpy as np
import scipy.sparse

from src.utils.quadrature import get_quadrature, TensorGrid
from .basis import TensorGrid3D


def assemble_diffusion_3d(batch, nx, ny, nz, config={}):
    """
        assemble_diffusion_3d(batch, nx, ny, nz, config={})

    3D equivalent of `assemble_diffusion`: create a finite volume discretisation of the 3D diffusion
    problem given by the first problem in the batch. Note that the batch should contain functions,
    not samples. As in 2D, the grid is a normal grid along the x-axis (whose boundaries have
    Dirichlet conditions) and a staggered grid along the y- and z-axes (whose boundaries have
    Neumann conditions):

    - x_i = i / (nx - 1), i = 1, 2, ..., nx - 2
    - y_j = (j - 0.5) / ny, j = 1, 2, ..., ny
    - z_k = (k - 0.5) / nz, k = 1, 2, ..., nz

    For a uniform grid, it should therefore hold that nx = ny + 1 = nz + 1. The nodes on the left
    and right boundaries are omitted since their values are known, so the discretisation produces a
    system of (nx - 2) * ny * nz equations.

    The output is a tuple (A, b, quad, ids):
    - A is a sparse symmetric positive definite matrix, and b a vector, such that if u = A⁻¹b, then
      reshape(u, (nx - 2, ny, nz))[i, j, k] is an approximation of u(x_{i+1}, y_j, z_k)
    - quad = (nodes_x, weights_x, nodes_y, weights_y, nodes_z, weights_z): the grid points, together
      with the weights of a quadrature rule on them
    - ids = (ix, iy, iz): the indices into nodes_x, nodes_y, nodes_z of the points that are present
      in the discretisation

    As in 2D, the Dirichlet boundary condition is θu = g, and setting config["powtheta"] replaces θ
    by θ^powtheta.
    """
    nodes_x, weights_x = get_quadrature(f"t{nx}")
    nodes_y, weights_y = get_quadrature(f"m{ny}")
    nodes_z, weights_z = get_quadrature(f"m{nz}")
    quad = (nodes_x, weights_x, nodes_y, weights_y, nodes_z, weights_z)
    grid = TensorGrid3D(nodes_x, nodes_y, nodes_z)
    theta = np.array(batch["theta"][0](grid).reshape(nx, ny, nz))
    f = np.array(batch["f"][0](grid).reshape(nx, ny, nz))
    g_left = np.array(batch["left:g"][0](TensorGrid(nodes_y, nodes_z)).reshape(ny, nz))
    g_right = np.array(batch["right:g"][0](TensorGrid(nodes_y, nodes_z)).reshape(ny, nz))
    eta_front = np.array(batch["front:eta"][0](TensorGrid(nodes_x, nodes_z)).reshape(nx, nz))
    eta_back = np.array(batch["back:eta"][0](TensorGrid(nodes_x, nodes_z)).reshape(nx, nz))
    eta_bottom = np.array(batch["bottom:eta"][0](TensorGrid(nodes_x, nodes_y)).reshape(nx, ny))
    eta_top = np.array(batch["top:eta"][0](TensorGrid(nodes_x, nodes_y)).reshape(nx, ny))

    # If "powtheta" is set, replace theta by theta ^ k
    if (pow := config.get("powtheta")) is not None:
        theta = theta**pow

    Dx = 1 / (nx - 1)
    Dy = 1 / ny
    Dz = 1 / nz

    # Coefficients of the couplings between neighbouring nodes along each axis, using the average
    # diffusivity of the two nodes. These are negative, and have one entry less than theta along the
    # corresponding axis.
    c_x = -0.5 * (theta[:-1, :, :] + theta[1:, :, :]) / Dx**2
    c_y = -0.5 * (theta[1:-1, :-1, :] + theta[1:-1, 1:, :]) / Dy**2
    c_z = -0.5 * (theta[1:-1, :, :-1] + theta[1:-1, :, 1:]) / Dz**2

    # The diagonal entries: minus the sum of the couplings to all neighbours. In the x-direction
    # this includes the couplings to the (omitted) Dirichlet boundary nodes. In the y- and
    # z-directions, the boundary faces have zero flux apart from the Neumann data
    diag = -(c_x[:-1] + c_x[1:])
    diag[:, :-1, :] -= c_y
    diag[:, 1:, :] -= c_y
    diag[:, :, :-1] -= c_z
    diag[:, :, 1:] -= c_z

    # Indices of the unknowns, with the z-index varying fastest
    shape = (nx - 2, ny, nz)
    idx = np.arange(np.prod(shape)).reshape(shape)
    neighbours = [
        (idx[:-1, :, :], idx[1:, :, :], c_x[1:-1]),
        (idx[:, :-1, :], idx[:, 1:, :], c_y),
        (idx[:, :, :-1], idx[:, :, 1:], c_z),
    ]
    rows = [idx.ravel()]
    cols = [idx.ravel()]
    vals = [diag.ravel()]
    for i, j, c in neighbours:
        # Add both (i, j) and (j, i) since A is symmetric
        rows += [i.ravel(), j.ravel()]
        cols += [j.ravel(), i.ravel()]
        vals += [c.ravel(), c.ravel()]

    N = idx.size
    A = scipy.sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)
    ).tocsr()

    b = f[1:-1, :, :].copy()
    # Dirichlet data (θu = g)
    b[0, :, :] -= c_x[0] * g_left / theta[0]
    b[-1, :, :] -= c_x[-1] * g_right / theta[-1]
    # Neumann data
    b[:, 0, :] += eta_front[1:-1, :] / Dy
    b[:, -1, :] += eta_back[1:-1, :] / Dy
    b[:, :, 0] += eta_bottom[1:-1, :] / Dz
    b[:, :, -1] += eta_top[1:-1, :] / Dz

    ids = (np.arange(1, nx - 1), np.arange(ny), np.arange(nz))
    return A, b.ravel(), quad, ids
