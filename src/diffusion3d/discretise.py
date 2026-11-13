# 3D equivalents of the functions in `src/discretise.py`.
#
# A 3D diffusion data set contains
# - 3D fields (theta, f, u), which are sampled on the tensor product of the x-, y-, and z-quadrature
#   rules
# - face data, which are 2D functions of the two coordinates tangential to the face:
#   - "left:g", "right:g" on x = 0 and x = 1, as functions of (y, z)
#   - "front:eta", "back:eta" on y = 0 and y = 1, as functions of (x, z)
#   - "bottom:eta", "top:eta" on z = 0 and z = 1, as functions of (x, y)
#
# As in 2D, the data on opposite faces is stacked, with the face index at the position of the
# normal axis:
# - "dirichlet:g": B × 2 × ny × nz
# - "neumann_y:eta": B × nx × 2 × nz
# - "neumann_z:eta": B × nx × ny × 2
from typing import Union

import torch

from src.dataset import DataSet
from src.training.subnets.basis import Functions
from src.utils import recursive_to
from src.utils.misc import BatchIterator, relative_errors
from src.utils.quadrature import TensorGrid, get_quadrature
from .basis import TensorGrid3D

AXES = ("x", "y", "z")

# For each face: the axis normal to the face (0, 1, 2 for x, y, z)
FACE_AXES = {
    "left": 0,
    "right": 0,
    "front": 1,
    "back": 1,
    "bottom": 2,
    "top": 2,
}

# Pairs of opposite faces, and the name of the stacked data
FACE_PAIRS = (
    ("left", "right", "dirichlet"),
    ("front", "back", "neumann_y"),
    ("bottom", "top", "neumann_z"),
)


def tangential_axes(face):
    normal = FACE_AXES[face]
    return tuple(axis for axis in range(3) if axis != normal)


def _get_quadratures(quad):
    # Normalise a quadrature specification to a dict {field: (nodes, weights)}. The entries can be
    # strings like "m100" or tuples of nodes and weights, and if a single quadrature rule is given
    # it is used for all fields
    if isinstance(quad, str) or isinstance(quad, tuple):
        quad = {"default": quad}

    return {
        field: get_quadrature(name) if isinstance(name, str) else name
        for (field, name) in quad.items()
    }


def _make_attributes_3d(
    fields,
    quad_x,
    quad_y,
    quad_z,
    device: Union[str, torch.device],
    dtype: Union[str, torch.dtype],
) -> dict:
    """
    3D equivalent of `_make_attributes`. `quad_y` and `quad_z` default to `quad_x` if they are
    `None`. Each quadrature argument is either a single quadrature rule (a string like "gl5x4" or a
    tuple of nodes and weights) or a dict mapping field names to quadrature rules, with a "default"
    entry.
    """
    quad_y = quad_x if quad_y is None else quad_y
    quad_z = quad_x if quad_z is None else quad_z
    quads = [_get_quadratures(q) for q in (quad_x, quad_y, quad_z)]

    def get_quad(axis, name):
        return quads[axis][name] if name in quads[axis] else quads[axis]["default"]

    attributes = {}
    for name, functions in fields.items():
        if not isinstance(functions, Functions):
            continue
        if functions.dim == 2:
            # Face data: use the quadrature rules of the two tangential axes. As in 2D, these are
            # looked up by the name of the field without the face, e.g. "g" for "left:g"
            face, qname = name.split(":", 1)
            for axis in tangential_axes(face):
                nodes, weights = get_quad(axis, qname)
                attributes[f"{name}/nodes_{AXES[axis]}"] = nodes
                attributes[f"{name}/weights_{AXES[axis]}"] = weights
        elif functions.dim == 3:
            for axis in range(3):
                nodes, weights = get_quad(axis, name)
                attributes[f"{name}/nodes_{AXES[axis]}"] = nodes
                attributes[f"{name}/weights_{AXES[axis]}"] = weights
        else:
            raise ValueError(f"unrecognised field {name} of dimension {functions.dim}")

    for axis in range(3):
        nodes, weights = quads[axis]["default"]
        attributes[f"nodes_{AXES[axis]}"] = nodes
        attributes[f"weights_{AXES[axis]}"] = weights

    attributes = recursive_to(attributes, device)
    attributes = recursive_to(attributes, dtype)
    return attributes


def _discretise_batch_3d(fields, attributes, dtype) -> dict:
    # Take a batch of functions, together with the attributes (containing quadrature rules)
    # and create a batch of discretised data
    B = len(fields["u"])
    batch = {}
    for name, field in fields.items():
        if not isinstance(field, Functions):
            continue

        if field.dim == 3:
            grid = TensorGrid3D(*(attributes[f"{name}/nodes_{axis}"] for axis in AXES))
            batch[name] = field(grid).reshape(B, *grid.grid_shape)
        elif field.dim == 2:
            face = name.split(":", 1)[0]
            nodes = [attributes[f"{name}/nodes_{AXES[axis]}"] for axis in tangential_axes(face)]
            grid = TensorGrid(*nodes)
            batch[name] = field(grid).reshape(B, *grid.grid_shape)

    # Stack the data on opposite faces, e.g. "left:g" and "right:g" into "dirichlet:g"
    for low, high, prefix in FACE_PAIRS:
        axis = FACE_AXES[low]
        for key in list(batch.keys()):
            if not key.startswith(low + ":"):
                continue
            field = key[len(low) + 1 :]
            if f"{high}:{field}" in batch:
                batch[f"{prefix}:{field}"] = torch.stack(
                    (batch[f"{low}:{field}"], batch[f"{high}:{field}"]), dim=1 + axis
                )

    batch = recursive_to(batch, dtype)
    return batch


def _concatenate_batches(batches):
    # Given an iterable of dicts with the same keys, concatenate the values per key
    result = None
    for batch in batches:
        if result is None:
            result = {key: [value] for (key, value) in batch.items()}
        else:
            for key, value in batch.items():
                result[key].append(value)

    return {key: torch.cat(value) for (key, value) in result.items()}


def discretise_3d(
    dataset,
    quad_x,
    quad_y=None,
    quad_z=None,
    device="cuda",
    batch_size=100,
    desc=None,
    dtype=torch.float32,
):
    """
    3D equivalent of `discretise`: evaluate all fields of the data set on the given quadrature
    rules. The resulting data set can be used to train a model.
    """
    attributes = _make_attributes_3d(dataset.data, quad_x, quad_y, quad_z, device, torch.float64)

    def batches():
        for indices in BatchIterator(len(dataset), batch_size, desc):
            batch_fun = dataset.get_batch(indices, device)
            batch = _discretise_batch_3d(batch_fun, attributes, dtype)
            yield recursive_to(batch, "cpu")

    return DataSet(
        _concatenate_batches(batches()),
        recursive_to(recursive_to(attributes, "cpu"), dtype),
        dataset.config,
        dataset.filename,
        dataset.part,
    )


def discretise_and_assemble_3d(
    dataset,
    model,
    quad_x,
    quad_y=None,
    quad_z=None,
    device="cuda",
    batch_size=100,
    desc=None,
    dtype=torch.float32,
):
    """
    3D equivalent of `discretise_and_assemble`: discretise the data set with the given quadrature
    rules, and assemble the inputs to the model's system network batch by batch.
    """
    model.to(device)
    model.eval()
    attributes = _make_attributes_3d(dataset.data, quad_x, quad_y, quad_z, device, torch.float64)
    attributes_dtype = recursive_to(attributes, dtype)

    def batches():
        for indices in BatchIterator(len(dataset), batch_size, desc):
            batch_fun = dataset.get_batch(indices, device)
            batch = _discretise_batch_3d(batch_fun, attributes, dtype)
            batch.update(attributes_dtype)
            batch["problem"] = batch_fun["problem"]
            with torch.no_grad():
                branch_batch = model.preassemble_batch(batch)

            branch_batch = recursive_to(branch_batch, "cpu")
            yield recursive_to(branch_batch, dtype)

    branch_data = _concatenate_batches(batches())
    branch_attributes = model.preassemble_attributes(attributes_dtype)
    branch_attributes = recursive_to(branch_attributes, "cpu")
    branch_attributes = recursive_to(branch_attributes, dtype)
    model.to("cpu")
    return DataSet(branch_data, branch_attributes, dataset.config, dataset.filename, dataset.part)


def discretise_and_test_3d(
    dataset: DataSet,
    model,
    *,
    quad_x,
    quad_y=None,
    quad_z=None,
    device,
    batch_size,
    desc=None,
    dtype=torch.float32,
):
    """
    3D equivalent of `discretise_and_test`: evaluate the model on the data set, returning the
    relative L² error per problem.
    """
    errs = []
    model.to(device)
    model.eval()
    attributes = _make_attributes_3d(
        dataset.data, quad_x, quad_y, quad_z, device, dtype=torch.float64
    )
    attributes = recursive_to(attributes, dtype)
    for indices in BatchIterator(len(dataset), batch_size, desc):
        batch_fun = dataset.get_batch(indices, device)
        batch = _discretise_batch_3d(batch_fun, attributes, dtype)
        batch.update(attributes)
        batch["problem"] = batch_fun["problem"]

        with torch.no_grad():
            u_pred = model(batch)

        err = relative_errors(
            u_pred,
            batch["u"],
            attributes["u/weights_x"],
            attributes["u/weights_y"],
            attributes["u/weights_z"],
        ).sqrt()
        errs.append(err)

    return torch.cat(errs)
