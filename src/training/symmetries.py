# This file contains a number of functions that are useful when dealing with the symmetries of the heat equation test problem:
# 1. Functions to create flipped versions of a batch of problems
# 2. Functions to handle symmetry group averaging in NGOs with B-spline bases
import torch
from copy import deepcopy
from math import isqrt


def _bspline_flip_vectors(vectors, flip_x, flip_y):
    if flip_x:
        vectors = vectors.flip(1)

    if flip_y:
        vectors = vectors.flip(2)

    return vectors


def bspline_expand_vectors(vectors, negate=""):
    """
        bspline_expand_vectors(vectors, negate="")

    Create 4 copies of the input, consisting of the original and flipped versions. `vectors` should
    be B × nx × ny, then output is B × 4 × nx × ny.

    Additionally, `negate` can be "" (default), "x", "y", or "xy", to also multiply the field by
    -1 when flipping in the "x"-direction, "y"-direction, or both. This is useful when mirroring
    a vector field, in which case the x-component should also be multiplied by -1 when mirroring
    horizontally, and the y-component should be multiplied by -1 when mirroring vertically.
    """
    assert (
        vectors.ndim == 3
    ), f"bspline_expand_vectors: expected 3-dimensional array, got shape {tuple(vectors.shape)}"

    neg_x = "x" in negate
    neg_y = "y" in negate
    vectors_00 = vectors
    vectors_10 = _bspline_flip_vectors(vectors, 1, 0) * (-1) ** neg_x
    vectors_01 = _bspline_flip_vectors(vectors, 0, 1) * (-1) ** neg_y
    vectors_11 = _bspline_flip_vectors(vectors, 1, 1) * (-1) ** (neg_x ^ neg_y)
    return torch.stack([vectors_00, vectors_10, vectors_01, vectors_11], 1)


def bspline_reduce_vectors(vectors, negate=""):
    """
        bspline_reduce_vectors(vectors, negate="")

    Reduce the input by flipping the inputs and averaging. `vectors` should be B × 4 × nx × ny, then
    output is B × nx × ny.

    Additionally, `negate` can be "" (default), "x", "y", or "xy", to also multiply the field by
    -1 when flipping in the "x"-direction, "y"-direction, or both. This is useful when mirroring
    a vector field, in which case the x-component should also be multiplied by -1 when mirroring
    horizontally, and the y-component should be multiplied by -1 when mirroring vertically.
    """
    assert (
        vectors.ndim == 4 and vectors.shape[1] == 4
    ), f"bspline_reduce_vectors: expected B × 4 × nx × ny array, got shape {tuple(vectors.shape)}"
    neg_x = "x" in negate
    neg_y = "y" in negate
    B = vectors.shape[0]
    vectors_00 = vectors[:, 0]
    vectors_10 = vectors[:, 1] * (-1) ** neg_x
    vectors_01 = vectors[:, 2] * (-1) ** neg_y
    vectors_11 = vectors[:, 3] * (-1) ** (neg_x ^ neg_y)
    return (
        vectors_00
        + _bspline_flip_vectors(vectors_10, 1, 0)
        + _bspline_flip_vectors(vectors_01, 0, 1)
        + _bspline_flip_vectors(vectors_11, 1, 1)
    ) / 4


def _bspline_flip_matrices(matrices, flip_x, flip_y):
    if flip_x:
        matrices = matrices.flip(1, 3)

    if flip_y:
        matrices = matrices.flip(2, 4)

    return matrices


def bspline_expand_matrices(matrices):
    """
        bspline_expand_matrices(matrices)

    Given an array of matrices of size B × nx × ny × nx × ny, permute it corresponding to mirroring
    in the x- and y-directions. The output will have shape B × 4 × nx × ny × nx × ny, corresponding
    to all four permutations stacked.
    """
    err = f"bspline_expand_matrices: expected shape B × nx × ny × nx × ny, got {tuple(matrices.shape)}"
    assert matrices.ndim == 5, err
    B, nx, ny, nx2, ny2 = matrices.shape
    assert nx == nx2 and ny == ny2, err
    matrices_00 = matrices
    matrices_10 = _bspline_flip_matrices(matrices, 1, 0)
    matrices_01 = _bspline_flip_matrices(matrices, 0, 1)
    matrices_11 = _bspline_flip_matrices(matrices, 1, 1)
    return torch.stack([matrices_00, matrices_10, matrices_01, matrices_11], 1)


def bspline_reduce_matrices(matrices):
    """
        bspline_reduce_matrices(matrices)

    Given an array of matrices of size B × 4 × nx × ny × nx × ny, perform symmetry group averaging.
    The output will have shape B × nx × ny × nx × ny, consisting of the four parts permuted and
    averaged.
    """
    err = f"bspline_reduce_matrices: expected shape B × 4 × nx × ny × nx × ny, got {tuple(matrices.shape)}"
    assert matrices.ndim == 6, err
    B, S, nx, ny, nx2, ny2 = matrices.shape
    assert S == 4 and nx == nx2 and ny == ny2, err
    matrices_00 = matrices[:, 0]
    matrices_10 = matrices[:, 1]
    matrices_01 = matrices[:, 2]
    matrices_11 = matrices[:, 3]
    return (
        matrices_00
        + _bspline_flip_matrices(matrices_10, 1, 0)
        + _bspline_flip_matrices(matrices_01, 0, 1)
        + _bspline_flip_matrices(matrices_11, 1, 1)
    ) / 4


def _batch_expand_matrix(m, negate=""):
    neg_x = "x" in negate
    neg_y = "y" in negate
    neg_xy = neg_x ^ neg_y
    m00 = m
    m10 = m.flip(1) * (-1) ** neg_x
    m01 = m.flip(2) * (-1) ** neg_y
    m11 = m.flip(1, 2) * (-1) ** neg_xy
    return torch.cat((m00, m10, m01, m11), dim=0)


def _batch_expand_matrix_grad(m):
    dx = m[:, 0, :, :]
    dy = m[:, 1, :, :]
    dx = _batch_expand_matrix(dx, "x")
    dy = _batch_expand_matrix(dy, "y")
    return torch.stack((dx, dy), dim=1)


def batch_expand(batch):
    """
    Given a batch of size B, return a batch of size 4B that also contains mirrored versions of the
    problems and solutions.
    """
    batch = dict(batch)
    batch["u"] = _batch_expand_matrix(batch["u"])
    batch["f"] = _batch_expand_matrix(batch["f"])
    batch["theta"] = _batch_expand_matrix(batch["theta"])
    batch["neumann:f"] = _batch_expand_matrix(batch["neumann:f"])
    batch["neumann:eta"] = _batch_expand_matrix(batch["neumann:eta"])
    batch["neumann:theta"] = _batch_expand_matrix(batch["neumann:theta"])
    batch["dirichlet:f"] = _batch_expand_matrix(batch["dirichlet:f"])
    batch["dirichlet:theta"] = _batch_expand_matrix(batch["dirichlet:theta"])
    batch["dirichlet:g"] = _batch_expand_matrix(batch["dirichlet:g"])
    batch["theta/grad"] = _batch_expand_matrix_grad(batch["theta/grad"])
    batch["u/grad"] = _batch_expand_matrix_grad(batch["u/grad"])

    if batch["problem"] == "advection-diffusion":
        batch["v_x"] = _batch_expand_matrix(batch["v_x"], "x")
        batch["v_y"] = _batch_expand_matrix(batch["v_y"], "y")
        batch["neumann:v_x"] = _batch_expand_matrix(batch["neumann:v_x"], "x")
        batch["neumann:v_y"] = _batch_expand_matrix(batch["neumann:v_y"], "y")
        batch["dirichlet:v_x"] = _batch_expand_matrix(batch["dirichlet:v_x"], "x")
        batch["dirichlet:v_y"] = _batch_expand_matrix(batch["dirichlet:v_y"], "y")

    return batch
