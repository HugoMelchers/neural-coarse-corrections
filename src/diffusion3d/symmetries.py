# 3D equivalents of the symmetry group averaging functions in `src/training/symmetries.py`.
#
# The 3D diffusion problem (Dirichlet on the x-faces, Neumann on the y- and z-faces) is invariant
# under mirroring in each of the x-, y-, and z-directions, giving a group of 8 symmetries. As in 2D,
# mirroring a B-spline basis on a symmetric knot vector corresponds to reversing the order of its
# basis functions, so the symmetries act on coefficient arrays by flipping axes.
from itertools import product

import torch

# All 8 combinations of (flip_x, flip_y, flip_z), starting with the identity
FLIPS_3D = list(product((False, True), repeat=3))


def _flip_dims(flips, offset):
    return [offset + axis for (axis, flip) in enumerate(flips) if flip]


def _flip_vectors(vectors, flips):
    # vectors: B × nx × ny × nz
    dims = _flip_dims(flips, 1)
    return vectors.flip(dims) if dims else vectors


def _flip_matrices(matrices, flips):
    # matrices: B × nx × ny × nz × nx × ny × nz, flipped on both the row and the column axes
    dims = _flip_dims(flips, 1) + _flip_dims(flips, 4)
    return matrices.flip(dims) if dims else matrices


def bspline_expand_vectors_3d(vectors):
    """
    Create 8 copies of the input, consisting of the original and mirrored versions. `vectors`
    should be B × nx × ny × nz, then the output is B × 8 × nx × ny × nz.
    """
    assert (
        vectors.ndim == 4
    ), f"bspline_expand_vectors_3d: expected 4-dimensional array, got shape {tuple(vectors.shape)}"
    return torch.stack([_flip_vectors(vectors, flips) for flips in FLIPS_3D], 1)


def bspline_reduce_vectors_3d(vectors):
    """
    Undo the mirroring of `bspline_expand_vectors_3d` and average. `vectors` should be
    B × 8 × nx × ny × nz, then the output is B × nx × ny × nz.
    """
    assert (
        vectors.ndim == 5 and vectors.shape[1] == 8
    ), f"bspline_reduce_vectors_3d: expected B × 8 × nx × ny × nz array, got {tuple(vectors.shape)}"
    return sum(_flip_vectors(vectors[:, s], flips) for (s, flips) in enumerate(FLIPS_3D)) / 8


def bspline_reduce_matrices_3d(matrices):
    """
    Given an array of matrices of size B × 8 × nx × ny × nz × nx × ny × nz, perform symmetry group
    averaging. The output has shape B × nx × ny × nz × nx × ny × nz.
    """
    err = (
        f"bspline_reduce_matrices_3d: expected B × 8 × (nx × ny × nz)², got {tuple(matrices.shape)}"
    )
    assert matrices.ndim == 8 and matrices.shape[1] == 8, err
    assert matrices.shape[2:5] == matrices.shape[5:8], err
    return sum(_flip_matrices(matrices[:, s], flips) for (s, flips) in enumerate(FLIPS_3D)) / 8
