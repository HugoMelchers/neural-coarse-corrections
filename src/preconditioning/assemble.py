import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator

from ..utils.quadrature import get_quadrature, TensorGrid


class StencilOperator(LinearOperator):
    def __init__(self, *, C, N, S, E, W, NE=None, NW=None, SE=None, SW=None):
        err_part1 = "StencilOperator: expected"

        # fmt: off
        assert C.ndim == 2, f"make_matrix_from_stencil: expected 2-dimensional array for C, got {C.ndim}-shape {C.shape}"
        nx, ny = C.shape
        assert N.shape == (nx, ny - 1), f"{err_part1} N.shape = C.shape - (0, 1), got {C.shape = }, {N.shape = }"
        assert S.shape == (nx, ny - 1), f"{err_part1} S.shape = C.shape - (0, 1), got {C.shape = }, {S.shape = }"
        assert E.shape == (nx - 1, ny), f"{err_part1} E.shape = C.shape - (1, 0), got {C.shape = }, {E.shape = }"
        assert W.shape == (nx - 1, ny), f"{err_part1} W.shape = C.shape - (1, 0), got {C.shape = }, {W.shape = }"

        if NE is not None:
            assert NE.shape == (nx - 1, ny - 1), f"{err_part1} NE.shape = C.shape - (1, 1), got {C.shape = }, {NE.shape = }"
        if SE is not None:
            assert SE.shape == (nx - 1, ny - 1), f"{err_part1} SE.shape = C.shape - (1, 1), got {C.shape = }, {SE.shape = }"
        if SW is not None:
            assert SW.shape == (nx - 1, ny - 1), f"{err_part1} SW.shape = C.shape - (1, 1), got {C.shape = }, {SW.shape = }"
        if NW is not None:
            assert NW.shape == (nx - 1, ny - 1), f"{err_part1} NW.shape = C.shape - (1, 1), got {C.shape = }, {NW.shape = }"
        # fmt: on

        self.nx = nx
        self.ny = ny
        self.shape = (nx * ny, nx * ny)

        self.C = C

        self.N = N
        self.S = S
        self.E = E
        self.W = W

        self.NE = NE
        self.NW = NW
        self.SE = SE
        self.SW = SW
        self.dtype = C.dtype

    def _matvec(self, x):
        x = np.reshape(x, (self.nx, self.ny))
        y = self.C * x
        y[:, :-1] += self.N * x[:, 1:]
        y[:, 1:] += self.S * x[:, :-1]
        y[:-1, :] += self.E * x[1:, :]
        y[1:, :] += self.W * x[:-1, :]

        if self.NE is not None:
            y[:-1, :-1] += self.NE * x[1:, 1:]
        if self.NW is not None:
            y[1:, :-1] += self.NW * x[:-1, 1:]
        if self.SE is not None:
            y[:-1, 1:] += self.SE * x[1:, :-1]
        if self.SW is not None:
            y[1:, 1:] += self.SW * x[:-1, :-1]

        return y.ravel()

    def _rmatvec(self, x):
        x = np.reshape(x, (self.nx, self.ny))
        y = self.C * x
        y[:, :-1] += self.S * x[:, 1:]
        y[:, 1:] += self.N * x[:, :-1]
        y[:-1, :] += self.W * x[1:, :]
        y[1:, :] += self.E * x[:-1, :]

        if self.SW is not None:
            y[:-1, :-1] += self.SW * x[1:, 1:]
        if self.SE is not None:
            y[1:, :-1] += self.SE * x[:-1, 1:]
        if self.NW is not None:
            y[:-1, 1:] += self.NW * x[1:, :-1]
        if self.NE is not None:
            y[1:, 1:] += self.NE * x[:-1, :-1]

        return y.ravel()

    def tosparse(self):
        def zeros(n):
            return np.zeros((n, 1))

        def vcat(*arrays):
            return np.concatenate(arrays, axis=0)

        def hcat(*arrays):
            return np.concatenate(arrays, axis=1)

        def strictsetdiag(A, vals, k):
            N = A.shape[0]
            assert vals.ndim == 1, f"got multidimensional shape {vals.shape}"
            absk = k if k > 0 else -k
            assert (
                vals.shape[0] == N - absk
            ), f"setting diag({k}) of {N}×{N} matrix needs {N - absk} values, got {vals.shape[0]}"
            A.setdiag(vals, k)

        nx, ny = self.nx, self.ny
        C = self.C
        N, S, E, W = self.N, self.S, self.E, self.W
        NE, NW, SE, SW = self.NE, self.NW, self.SE, self.SW
        A = scipy.sparse.lil_matrix((nx * ny, nx * ny), dtype=self.C.dtype)
        strictsetdiag(A, C.flatten(), 0)
        strictsetdiag(A, hcat(N, zeros(nx)).flatten()[:-1], 1)
        strictsetdiag(A, hcat(zeros(nx), S).flatten()[1:], -1)
        strictsetdiag(A, E.flatten(), ny)
        strictsetdiag(A, W.flatten(), -ny)

        if NE is not None:
            strictsetdiag(A, hcat(NE, zeros(nx - 1)).flatten()[:-1], ny + 1)

        if SE is not None:
            strictsetdiag(
                A,
                hcat(zeros(nx), vcat(SE, zeros(ny - 1).T)).flatten()[: -(ny - 1)],
                ny - 1,
            )

        if SW is not None:
            strictsetdiag(A, hcat(zeros(nx - 1), SW).flatten()[1:], -(ny + 1))

        if NW is not None:
            strictsetdiag(
                A,
                hcat(
                    vcat(zeros(ny - 1).T, NW),
                    zeros(nx),
                ).flatten()[ny - 1 :],
                -(ny - 1),
            )

        return A.tocsr()


def assemble(batch, nx, ny):
    if batch["problem"] == "diffusion":
        return assemble_diffusion(batch, nx, ny)
    elif batch["problem"] == "advection-diffusion":
        return assemble_advection_diffusion(batch, nx, ny)
    elif batch["problem"] == "helmholtz":
        return assemble_helmholtz(batch, nx, ny)
    else:
        raise ValueError(f"assemble(): got unsupported problem type \"{batch['problem']}\"")


def assemble_diffusion(batch, nx, ny):
    """
        assemble_diffusion(batch, nx, ny)

    Create a finite volume discretisation of the diffusion problem given by the first problem in
    the batch. Note that the batch should contain functions, not samples. The discretisation will
    produce a system of (nx - 2) * ny equations (i.e. the left and right boundary are omitted due to
    the boundary conditions). It uses a normal grid along the x-axis, and a staggered grid along the
    y-axis, i.e.:

    - x_i = i / (nx - 1), i = 1, 2, ..., nx - 2
    - y_j = (j - 0.5) / ny, j = 1, 2, ..., ny

    This means that for a uniformly spaced grid, i.e. with Dx = Dy, it should be that nx = ny + 1.

    The output is a pair (A, b) so that if u = A⁻¹b, then reshape(u, nx - 2, ny)[i, j] is an
    approximation for u(x_i, y_j). Note that the matrix A is symmetric positive definite.
    """
    # assert nx == ny + 1
    nodes_x, weights_x = get_quadrature(f"t{nx}")
    nodes_y, weights_y = get_quadrature(f"m{ny}")
    quad = (nodes_x, weights_x, nodes_y, weights_y)
    grid = TensorGrid(nodes_x, nodes_y)
    theta = np.array(batch["theta"][0](grid).reshape(nx, ny))
    f = np.array(batch["f"][0](grid).reshape(nx, ny))
    eta_bottom = np.array(batch["bottom:eta"][0](nodes_x))
    eta_top = np.array(batch["top:eta"][0](nodes_x))
    g_left = np.array(batch["left:g"][0](nodes_y))
    g_right = np.array(batch["right:g"][0](nodes_y))

    eta = np.stack((eta_bottom, eta_top), axis=-1)
    g = np.stack((g_left, g_right), axis=0)
    Dx = 1 / (nx - 1)
    Dy = 1 / ny

    theta_avg_x = 0.5 * (theta[:-1, :] + theta[1:, :])
    theta_avg_y = 0.5 * (theta[:, :-1] + theta[:, 1:])
    a_E = -theta_avg_x[1:, :] / Dx**2
    a_W = -theta_avg_x[:-1, :] / Dx**2
    a_N = -theta_avg_y[1:-1, :] / Dy**2
    a_S = -theta_avg_y[1:-1, :] / Dy**2
    a_C = -a_E - a_W
    a_C[:, :-1] -= a_N
    a_C[:, 1:] -= a_S

    A = StencilOperator(C=a_C, N=a_N, S=a_S, E=a_E[:-1, :], W=a_W[1:, :]).tosparse()
    b = f[1:-1, :]
    # Dirichlet data
    b[0, :] -= a_W[0, :] * g[0, :] / theta[0, :]
    b[-1, :] -= a_E[-1, :] * g[1, :] / theta[-1, :]
    # Neumann data
    b[:, 0] += eta[1:-1, 0] / Dy
    b[:, -1] += eta[1:-1, 1] / Dy
    return A, b.flatten(), quad, (np.arange(1, nx - 1), np.arange(ny))


def assemble_advection_diffusion(batch, nx, ny):
    """
        assemble_advection_diffusion(batch, nx, ny)

    Assemble a linear system Au = b corresponding to a finite-volume discretisation of the
    advection-diffusion equation with the given parameters. The batch should contain the following
    functions:
    - theta
    - v_x, v_y
    - f
    - left:g, right:g
    - bottom:eta, top:eta

    The PDE is discretised on a normal grid along the x and y axes, i.e.

    - x_i = i / (nx - 1), i = 1, 2, ..., nx - 2
    - y_j = j / (ny - 1), j = 0, 1, ..., ny - 1

    Note that the points on the left and right boundaries are not part of the discretisation (since
    their values are known), so the resulting PDE is over (nx - 2) * ny variables.
    """
    # Construct quadrature rules
    nodes_x, weights_x = get_quadrature(f"t{nx}")
    nodes_y, weights_y = get_quadrature(f"t{ny}")
    quad = (nodes_x, weights_x, nodes_y, weights_y)
    grid = TensorGrid(nodes_x, nodes_y)

    # Sample functions
    theta = np.array(batch["theta"][0](grid).reshape(nx, ny))
    v_x = np.array(batch["v_x"][0](grid).reshape(nx, ny))
    v_y = np.array(batch["v_y"][0](grid).reshape(nx, ny))
    f = np.array(batch["f"][0](grid).reshape(nx, ny))

    eta_bottom = np.array(batch["bottom:eta"][0](nodes_x))
    eta_top = np.array(batch["top:eta"][0](nodes_x))
    g_left = np.array(batch["left:g"][0](nodes_y))
    g_right = np.array(batch["right:g"][0](nodes_y))

    eta = np.stack((eta_bottom, eta_top), axis=-1)
    g = np.stack((g_left, g_right), axis=0)

    Dx = 1 / (nx - 1)
    Dy = 1 / (ny - 1)

    def fluxB(z):
        return z / np.expm1(z)

    def fluxW(z):
        # The function W(z) satisfies W(z) + W(-z) = 1
        # Numerically, however, W is easier to evaluate for large negative z than for large positive z
        # This is because expm1(large) returns inf, whereas expm1(-large) = -1
        # So, for positive z we compute W(z) as W(z) = 1 - W(-z)
        return np.where(
            np.abs(z) < 1e-5,
            1 / 2 - z / 12 + z**3 / 720 - z**5 / 30240,
            np.where(
                z <= 0,
                (np.expm1(z) - z) / (z * np.expm1(z)),
                1 - (np.expm1(-z) + z) / (-z * np.expm1(-z)),
            ),
        )

    # Directions:
    # North (positive y)
    # South (negative y)
    # East  (positive x)
    # West  (negative x)
    # The east and west boundaries are omitted in the FD discretisation
    lambda_x = v_x / theta
    lambda_y = v_y / theta
    Px = lambda_x * Dx
    Py = lambda_y * Dy
    # Some of the variables defined here are given on staggered grids
    # The variables with a subscript _x are given on x-staggered grids, i.e. (i + 1/2, j)
    # These have shape (n - 1) × n (except v_x and lambda_x)
    # The variables with a subscript _y are given on y-staggered grids, i.e. (i, j + 1/2)
    # These have shape n × (n - 1) (except v_y and lambda_y)
    lambda_bar_x = (lambda_x[:-1, :] + lambda_x[1:, :]) / 2
    lambda_bar_y = (lambda_y[:, :-1] + lambda_y[:, 1:]) / 2
    P_bar_x = (Px[:-1, :] + Px[1:, :]) / 2
    P_bar_y = (Py[:, :-1] + Py[:, 1:]) / 2
    W_minus_x = fluxW(-P_bar_x)
    W_plus_x = fluxW(P_bar_x)
    W_minus_y = fluxW(-P_bar_y)
    W_plus_y = fluxW(P_bar_y)
    lambda_tilde_x = W_minus_x * lambda_x[:-1, :] + W_plus_x * lambda_x[1:, :]
    lambda_tilde_y = W_minus_y * lambda_y[:, :-1] + W_plus_y * lambda_y[:, 1:]
    theta_tilde_x = W_minus_x * theta[:-1, :] + W_plus_x * theta[1:, :]
    theta_tilde_y = W_minus_y * theta[:, :-1] + W_plus_y * theta[:, 1:]
    Ex = lambda_tilde_x / lambda_bar_x * theta_tilde_x
    Ey = lambda_tilde_y / lambda_bar_y * theta_tilde_y
    B_minus_x = fluxB(-P_bar_x)
    B_plus_x = fluxB(P_bar_x)
    B_minus_y = fluxB(-P_bar_y)
    B_plus_y = fluxB(P_bar_y)
    alpha_x = Ex / Dx * B_minus_x
    beta_x = Ex / Dx * B_plus_x
    alpha_y = Ey / Dy * B_minus_y
    beta_y = Ey / Dy * B_plus_y
    gamma_x = np.maximum(0.5 - W_plus_x, 0.0)
    gamma_y = np.maximum(0.5 - W_plus_y, 0.0)
    delta_x = np.minimum(0.5 - W_plus_x, 0.0)
    delta_y = np.minimum(0.5 - W_plus_y, 0.0)

    # Compute coefficients of horizontal fluxes
    # First for the interior
    # Then using a simplified rule near the Neumann/Robin boundaries
    cx_W = alpha_x[:, 1:-1] - Dx / Dy * gamma_x[:, 1:-1] * (alpha_y[:-1, 1:] + beta_y[:-1, :-1])
    cx_E = -beta_x[:, 1:-1] - Dx / Dy * delta_x[:, 1:-1] * (alpha_y[1:, 1:] + beta_y[1:, :-1])
    cx_NW = Dx / Dy * beta_y[:-1, 1:] * gamma_x[:, 1:-1]
    cx_SW = Dx / Dy * alpha_y[:-1, :-1] * gamma_x[:, 1:-1]
    cx_NE = Dx / Dy * beta_y[1:, 1:] * delta_x[:, 1:-1]
    cx_SE = Dx / Dy * alpha_y[1:, :-1] * delta_x[:, 1:-1]
    dx_W = -Dx * gamma_x[:, :]
    dx_E = -Dx * delta_x[:, :]

    def hcat(*arrays):
        arrays = [array if array.ndim == 2 else array[:, None] for array in arrays]
        return np.concatenate(arrays, axis=1)

    # The cx_ coefficients are now (n - 1) × (n - 2). We need to extend this to (n - 1) × n
    # We do this by using the simple 1-dimensional flux formulations (i.e. excluding the crosswind-diffusion terms)

    cx_W = hcat(alpha_x[:, 0], cx_W, alpha_x[:, -1])
    cx_E = hcat(-beta_x[:, 0], cx_E, -beta_x[:, -1])

    zs = np.zeros(nx - 1)
    cx_NW = hcat(zs, cx_NW, zs)
    cx_SW = hcat(zs, cx_SW, zs)
    cx_NE = hcat(zs, cx_NE, zs)
    cx_SE = hcat(zs, cx_SE, zs)

    # Compute coefficients of vertical fluxes
    # Now, we only have the Dirichlet boundary conditions to deal with
    cy_S = alpha_y[1:-1, :] - Dy / Dx * gamma_y[1:-1, :] * (alpha_x[1:, :-1] + beta_x[:-1, :-1])
    cy_N = -beta_y[1:-1, :] - Dy / Dx * delta_y[1:-1, :] * (alpha_x[1:, 1:] + beta_x[:-1, 1:])
    cy_SE = Dy / Dx * beta_x[1:, :-1] * gamma_y[1:-1, :]
    cy_SW = Dy / Dx * alpha_x[:-1, :-1] * gamma_y[1:-1, :]
    cy_NE = Dy / Dx * beta_x[1:, 1:] * delta_y[1:-1, :]
    cy_NW = Dy / Dx * alpha_x[:-1, 1:] * delta_y[1:-1, :]
    dy_S = -Dy * gamma_y[1:-1, :]
    dy_N = -Dy * delta_y[1:-1, :]

    # Compute coefficients in matrix and right-hand-side operator
    # For the middle nodes (i.e. excluding the bottom and top boundaries), these are derived from the bordering fluxes
    # For the bottom and top nodes, they include only one vertical flux term, the other one given by the boundary condition
    # The horizontal fluxes have shape (n - 1, n)
    # The vertical fluxes have shape (n - 2, n + 1)
    zs = np.zeros(nx - 2)
    cy_S = hcat(zs, cy_S, zs)
    cy_SE = hcat(zs, cy_SE, zs)
    cy_SW = hcat(zs, cy_SW, zs)

    cy_N = hcat(zs, cy_N, zs)
    cy_NE = hcat(zs, cy_NE, zs)
    cy_NW = hcat(zs, cy_NW, zs)

    dy_S = hcat(zs, dy_S, zs)
    dy_N = hcat(zs, dy_N, zs)

    # For the vertical fluxes, we pad with zeros on the top (for the north dependencies) or bottom (for the south dependencies)
    # These values will instead be given by the boundary conditions

    # Add scaling for the bottom and top rows of cells, which are half as tall
    m = np.ones((1, ny))
    m[:, 0] *= 2
    m[:, -1] *= 2
    a_C = (cx_W[1:, :] - cx_E[:-1, :]) / Dx + (cy_S[:, 1:] - cy_N[:, :-1]) / Dy * m
    a_N = (cx_NW[1:, :] - cx_NE[:-1, :]) / Dx + cy_N[:, 1:] / Dy * m
    a_E = cx_E[1:, :] / Dx + (cy_SE[:, 1:] - cy_NE[:, :-1]) / Dy * m
    a_S = (cx_SW[1:, :] - cx_SE[:-1, :]) / Dx - cy_S[:, :-1] / Dy * m
    a_W = -cx_W[:-1, :] / Dx + (cy_SW[:, 1:] - cy_NW[:, :-1]) / Dy * m

    a_NE = cx_NE[1:, :] / Dx + cy_NE[:, 1:] / Dy * m
    a_SE = cx_SE[1:, :] / Dx - cy_SE[:, :-1] / Dy * m
    a_SW = -cx_SW[:-1, :] / Dx - cy_SW[:, :-1] / Dy * m
    a_NW = -cx_NW[:-1, :] / Dx + cy_NW[:, 1:] / Dy * m

    # For the Neumann boundary (i.e. where v.n > 0), we have to update a_C
    # If v > 0 this is the top boundary
    # If v < 0 this is the bottom boundary
    a_C[:, -1] += 2 / Dy * np.maximum(v_y[1:-1, -1], 0.0)
    a_C[:, 0] += 2 / Dy * np.maximum(-v_y[1:-1, 0], 0.0)

    A = StencilOperator(
        C=a_C,
        N=a_N[:, :-1],
        S=a_S[:, 1:],
        E=a_E[:-1, :],
        W=a_W[1:, :],
        NE=a_NE[:-1, :-1],
        NW=a_NW[1:, :-1],
        SE=a_SE[:-1, 1:],
        SW=a_SW[1:, 1:],
    ).tosparse()

    # Compute right-hand-side contribution factors excluding the bottom and top boundary
    # In the interior (i.e. excluding the top and bottom boundaries), we already know what to do
    b_C = 1 + (dx_W[1:, :] - dx_E[:-1, :]) / Dx + (dy_S[:, 1:] - dy_N[:, :-1]) / Dy
    b_N = dy_N[:, 1:] / Dy
    b_E = dx_E[1:, :] / Dx
    b_S = -dy_S[:, :-1] / Dy
    b_W = -dx_W[:-1, :] / Dx
    ss = f
    b = b_C * ss[1:-1, :]
    b[:, 1:-1] += b_N[:, 1:-1] * ss[1:-1, 2:]
    b[:, 1:-1] += b_S[:, 1:-1] * ss[1:-1, :-2]
    b[:, 1:-1] += b_E[:, 1:-1] * ss[2:, 1:-1]
    b[:, 1:-1] += b_W[:, 1:-1] * ss[:-2, 1:-1]
    # For the bottom and top boundaries we use the modified scheme
    b[:, 0] = 0.75 * ss[1:-1, 0] + 0.25 * ss[1:-1, 1] + 2 / Dy * eta[1:-1, 0]
    b[:, -1] = 0.75 * ss[1:-1, -1] + 0.25 * ss[1:-1, -2] + 2 / Dy * eta[1:-1, -1]

    g_E = g[-1, :] / theta[-1, :]
    g_W = g[0, :] / theta[0, :]
    b[-1, :-1] -= a_NE[-1, :-1] * g_E[1:]
    b[-1, :] -= a_E[-1, :] * g_E[:]
    b[-1, 1:] -= a_SE[-1, 1:] * g_E[:-1]

    b[0, :-1] -= a_NW[0, :-1] * g_W[1:]
    b[0, :] -= a_W[0, :] * g_W[:]
    b[0, 1:] -= a_SW[0, 1:] * g_W[:-1]

    b = b.flatten()
    return A, b, quad, (np.arange(1, nx - 1), np.arange(ny))


def assemble_helmholtz(batch, nx, ny):
    """
        assemble_helmholtz(batch, nx, ny)

    Create a finite difference discretisation of the Helmholtz problem given by the first problem in
    the batch. Note that the batch should contain functions, not samples. The discretisation will
    produce a system of (nx - 2) * ny equations (i.e. the left and right boundary are omitted due to
    the boundary conditions). It uses normal grids along the x- and y-axes, i.e.:

    - x_i = i / (nx - 1), i = 1, 2, ..., nx - 2
    - y_j = j / (ny - 1) / ny, j = 0, 1, 2, ..., ny - 1

    This means that for a uniformly spaced grid, i.e. with Dx = Dy, it should be that nx = ny.

    The output is a pair (A, b) so that if u = A⁻¹b, then reshape(u, nx - 2, ny)[i, j] is an
    approximation for u(x_i, y_j). Note that A is complex and symmetric but not Hermitian. (TODO check this)
    """
    nodes_x, weights_x = get_quadrature(f"t{nx}")
    nodes_y, weights_y = get_quadrature(f"t{ny}")
    quad = (nodes_x, weights_x, nodes_y, weights_y)
    grid = TensorGrid(nodes_x, nodes_y)
    theta = np.array(batch["theta"][0](grid).reshape(nx, ny))
    f = np.array(batch["f"][0](grid).reshape(nx, ny))
    eta_bottom = np.array(batch["bottom:eta"][0](nodes_x))
    eta_top = np.array(batch["top:eta"][0](nodes_x))
    g_left = np.array(batch["left:g"][0](nodes_y))
    g_right = np.array(batch["right:g"][0](nodes_y))

    eta_bottomtop = np.stack((eta_bottom, eta_top), axis=-1)
    g_leftright = np.stack((g_left, g_right), axis=0)
    Dx = 1 / (nx - 1)
    Dy = 1 / (ny - 1)

    a_E = np.full((nx - 3, ny), -1 / Dx**2, dtype=np.complex128)
    a_W = np.full((nx - 3, ny), -1 / Dx**2, dtype=np.complex128)
    a_N = np.full((nx - 2, ny - 1), -1 / Dy**2, dtype=np.complex128)
    a_S = np.full((nx - 2, ny - 1), -1 / Dy**2, dtype=np.complex128)
    a_C = np.array(2 / Dx**2 + 2 / Dy**2 - theta[1:-1, :] ** 2, dtype=np.complex128)

    # Handle impedance boundary conditions
    a_C[:, 0] -= 2j * theta[1:-1, 0] / Dy
    a_C[:, -1] -= 2j * theta[1:-1, -1] / Dy
    a_N[:, 0] *= 2
    a_S[:, -1] *= 2

    A = StencilOperator(
        C=a_C,
        N=a_N,
        S=a_S,
        E=a_E,
        W=a_W,
    ).tosparse()
    b = f[1:-1, :]
    # Dirichlet data
    b[0, :] += g_leftright[0, :] / Dx**2
    b[-1, :] += g_leftright[1, :] / Dx**2
    b[:, 0] += eta_bottomtop[1:-1, 0] * 2 / Dy
    b[:, -1] += eta_bottomtop[1:-1, 1] * 2 / Dy
    return A, b.flatten(), quad, (np.arange(1, nx - 1), np.arange(ny))
