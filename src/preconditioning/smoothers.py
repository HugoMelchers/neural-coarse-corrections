import scipy
from scipy.sparse.linalg import LinearOperator, splu
from scipy.special import kn as besselk, y0, y0_zeros
import numpy as np
import torch
import ilupp


def _filter_conv(nx, ny, filter, x):
    w2 = filter.shape[-1]
    w = (w2 - 1) // 2
    x = torch.tensor(x, dtype=torch.float64).reshape(1, nx, ny)
    y = torch.nn.functional.conv2d(x, filter, bias=None, stride=1, padding=w)
    return np.array(y.ravel())


def _filter_fft_prepare(nx, ny, filter):
    # pad to nx + 2w by ny + 2w
    w2 = filter.shape[-1]
    w = (w2 - 1) // 2
    filter_padded = np.zeros((nx + 2 * w, ny + 2 * w), dtype=filter.dtype)
    filter_padded[: w + 1, : w + 1] = filter[w:, w:]
    filter_padded[: w + 1, -w:] = filter[w:, :w]
    filter_padded[-w:, : w + 1] = filter[:w, w:]
    filter_padded[-w:, -w:] = filter[:w, :w]
    filter_fft = scipy.fft.fft2(filter_padded)
    return filter_fft.conj()


def _filter_fft(nx, ny, filter, x, filter_fft=None):
    w2 = filter.shape[-1]
    w = (w2 - 1) // 2
    if filter_fft is None:
        filter_fft = _filter_fft_prepare(nx, ny, filter)

    x = np.reshape(x, (nx, ny))
    x_padded = np.zeros((nx + 2 * w, ny + 2 * w), dtype=x.dtype)
    x_padded[w:-w, w:-w] = x
    x_fft = scipy.fft.fft2(x_padded)
    x_fft *= filter_fft
    y_padded = scipy.fft.ifft2(x_fft)

    def iscomplex(r):
        return r.dtype == np.complex128

    if iscomplex(x) or iscomplex(filter):
        return y_padded[w:-w, w:-w].ravel()
    else:
        return y_padded.real[w:-w, w:-w].ravel()


class JacobiSmoother(LinearOperator):
    def __init__(self, A):
        self.dtype = A.dtype
        self.shape = A.shape
        self.invD = 1 / A.diagonal()

    def _matvec(self, x):
        return x * self.invD


class BlockJacobiSmoother(LinearOperator):
    def __init__(self, A, nx, ny, Bx, By):
        self.dtype = A.dtype
        self.shape = A.shape
        # First, extract the diagonal blocks whose inverse we use.
        # The blocks are based on rectangular (approximately square) subdomains of the (0, 1)²
        # domain.
        # Boundaries of blocks in the x- and y-directions
        bxs = np.round(np.linspace(0, nx, Bx + 1)).astype(np.int32)
        bys = np.round(np.linspace(0, ny, By + 1)).astype(np.int32)
        bxs = [np.arange(bxs[i - 1], bxs[i]) for i in range(1, len(bxs))]
        bys = [np.arange(bys[i - 1], bys[i]) for i in range(1, len(bys))]
        # Now compute the indices into A per block. A is ordered y-varies-first, so the point (i, j)
        # becomes the index ny * i + j.
        block_ids = [(ny * bx[:, None] + by[None, :]).ravel() for by in bys for bx in bxs]
        blocks = [A[i][:, i].tocsc() for i in block_ids]
        self.inv_blocks = [splu(block) for block in blocks]
        self.block_ids = block_ids

    def _matvec(self, x):
        x_blocks = [x[ids] for ids in self.block_ids]
        y_blocks = [block.solve(x_block) for (block, x_block) in zip(self.inv_blocks, x_blocks)]
        y = np.zeros(x.shape, dtype=x.dtype)
        for ids, block in zip(self.block_ids, y_blocks):
            y[ids] = block

        return y


class DiffusionSmoother(LinearOperator):
    def __init__(self, h, w, theta, r):
        nx, ny = theta.shape
        self.nx = nx
        self.ny = ny
        t = np.linspace(-h * w, h * w, 2 * w + 1)
        xs = t[:, None]
        ys = t[None, :]
        dist = np.hypot(xs, ys)
        dist[w, w] = 1.0  # set the zero value to something else to avoid numpy giving a warning

        H = (w + 1) * h
        filt = (-(h**2) / (2 * np.pi)) * np.log(dist / H)
        filt[filt < 0] = 0

        if w > 0:
            # Set middle coefficient, so that G applied to the stencil produces r in the middle
            # part, i.e.:
            #   [  0    -1/h²     0  ]       [ . . . ]
            #   [-1/h²   4/h²   -1/h²] * G = [ . r . ]
            #   [  0    -1/h²     0  ]       [ . . . ]
            # This means that
            #
            #   G_00 = h²/4 * (r + 1/h² G_01 + ...)
            #
            # Note that the four neighbouring coefficients of G and the stencil are the same, so
            #
            #   G_00 = h²/4 * (r + 4/h² G_01) = h² r / 4 + G_01
            filt[w, w] = h**2 * r / 4 + filt[w, w + 1]

        self.filt = filt
        self.filt_fft = _filter_fft_prepare(nx, ny, self.filt)
        self.dtype = np.float64
        self.shape = (nx * ny, nx * ny)
        r = 1 / np.sqrt(theta).ravel()
        self.scale_pre = r
        self.scale_post = r

    def _matvec(self, x):
        if x.ndim == 2:
            x = x.flatten()
        x = x * self.scale_pre
        x = _filter_fft(self.nx, self.ny, self.filt, x, self.filt_fft)
        x = x * self.scale_post
        return x


class DownwindGS(LinearOperator):
    def __init__(self, xs, ys, A, vx, vy):
        xs = xs[:, None]
        ys = ys[None, :]
        D = (vx * xs + vy * ys).ravel()
        P = np.argsort(D)
        Pi = np.zeros(P.size, dtype=P.dtype)
        Pi[P] = np.arange(P.size)
        A_perm = A[P, :][:, P].tocsc()
        n = A.shape[0]
        self.P = P
        self.Pi = Pi
        self.dtype = A.dtype
        self.shape = A.shape
        self.AP_lo = scipy.sparse.tril(A_perm).tocsr()

    def _matvec(self, x):
        if x.ndim == 2:
            x = x.flatten()
        # 1. Permute
        x = x[self.P]
        # 2. Do block triangular solve
        x = scipy.sparse.linalg.spsolve_triangular(self.AP_lo, x, lower=True)
        return x[self.Pi]
