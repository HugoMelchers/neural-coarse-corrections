from typing import Tuple, Union

import torch

from .utils import recursive_to
from .utils.misc import BatchIterator, relative_errors, norm_sq
from .utils.quadrature import TensorGrid, tensorised, get_quadrature
from .training.subnets.basis import Functions
from .dataset import DataSet


def _make_attributes(
    fields,
    quad_x,
    quad_y,
    device: Union[str, torch.device],
    dtype: Union[str, torch.dtype],
) -> dict:
    """
    Given a 1D quadrature rule (nodes, weights), construct dataset attributes that use this
    quadrature rule for the boundaries, and its tensor product for the interior.
    """
    if quad_y is None:
        quad_y = quad_x

    if isinstance(quad_x, str) or isinstance(quad_x, tuple):
        # If a single quadrature rule is given, use that for all fieldss
        quad_x = {"default": quad_x}

    if isinstance(quad_y, str) or isinstance(quad_y, tuple):
        quad_y = {"default": quad_y}

    # Actually get the quadrature rules. The entries of quad can be strings like "m100"
    # corresponding to a 100-point midpoint discretisation, or a tuple of nodes and weights
    quad_x = {
        field: get_quadrature(name) if isinstance(name, str) else name
        for (field, name) in quad_x.items()
    }
    quad_y = {
        field: get_quadrature(name) if isinstance(name, str) else name
        for (field, name) in quad_y.items()
    }

    attributes = {}
    for name, functions in fields.items():
        if functions.dim == 1:
            # Either use the x or y quadrature, depending on whether this is Dirichlet or Neumann
            # boundary data
            if name.startswith("bottom:") or name.startswith("top:"):
                qname = name[name.index(":") + 1 :]
                quad_i = quad_x[qname] if qname in quad_x else quad_x["default"]
            elif name.startswith("left:") or name.startswith("right:"):
                qname = name[name.index(":") + 1 :]
                quad_i = quad_y[qname] if qname in quad_y else quad_y["default"]
            else:
                raise ValueError(f"unrecognised field {name}")

            attributes[name + "/nodes"] = quad_i[0]
            attributes[name + "/weights"] = quad_i[1]
        else:
            quad_x_i = quad_x[name] if name in quad_x else quad_x["default"]
            quad_y_i = quad_y[name] if name in quad_y else quad_y["default"]
            attributes[name + "/nodes_x"] = quad_x_i[0]
            attributes[name + "/weights_x"] = quad_x_i[1]
            attributes[name + "/nodes_y"] = quad_y_i[0]
            attributes[name + "/weights_y"] = quad_y_i[1]

    attributes["nodes_x"] = quad_x["default"][0]
    attributes["nodes_y"] = quad_y["default"][0]
    attributes["weights_x"] = quad_x["default"][1]
    attributes["weights_y"] = quad_y["default"][1]

    attributes = recursive_to(attributes, device)
    attributes = recursive_to(attributes, dtype)
    return attributes


def _discretise_batch(fields, attributes, dtype) -> dict:
    # Take a batch of functions, together with the attributes (containing quadrature rules)
    # and create a batch of discretised data
    B = len(fields["u"])
    batch = {}
    on_edges = {"left": [], "right": [], "bottom": [], "top": []}
    for name, field in fields.items():
        if not isinstance(field, Functions):
            continue
        for edge in on_edges:
            if name.startswith(edge + ":"):
                on_edges[edge].append(name[1 + len(edge) :])

        # Look up quadrature nodes
        if field.dim == 1:
            nodes = attributes[name + "/nodes"]
            batch[name] = field(nodes)
            batch[name + "/deriv"] = field.derivative(nodes)
        elif field.dim == 2:
            nodes_x = attributes[name + "/nodes_x"]
            nodes_y = attributes[name + "/nodes_y"]
            grid = TensorGrid(nodes_x, nodes_y)
            batch[name] = field(grid).reshape(B, *grid.grid_shape)
            batch[name + "/grad"] = field.gradient(grid).reshape(B, 2, *grid.grid_shape)
            # Also sample 2D fields on the boundaries
            zero = torch.tensor([0.0], dtype=nodes_x.dtype, device=nodes_x.device)
            one = torch.tensor([1.0], dtype=nodes_x.dtype, device=nodes_x.device)
            batch["left:" + name] = field(TensorGrid(zero, nodes_y))
            batch["right:" + name] = field(TensorGrid(one, nodes_y))
            batch["bottom:" + name] = field(TensorGrid(nodes_x, zero))
            batch["top:" + name] = field(TensorGrid(nodes_x, one))
            # Make sure that a field like dirichlet/theta is also available, containing the samples
            # on the left and right boundaries together
            on_edges["left"].append(name)
            on_edges["right"].append(name)
            on_edges["bottom"].append(name)
            on_edges["top"].append(name)

    # As a final step, concatenate left and right fields and bottom and top fields
    for field in on_edges["left"]:
        if field not in on_edges["right"]:
            continue

        batch["dirichlet:" + field] = torch.stack(
            (batch["left:" + field], batch["right:" + field]), dim=1
        )

    for field in on_edges["bottom"]:
        if field not in on_edges["top"]:
            continue

        batch["neumann:" + field] = torch.stack(
            (batch["bottom:" + field], batch["top:" + field]), dim=2
        )

    batch = recursive_to(batch, dtype)
    return batch


def discretise(
    dataset, quad_x, quad_y=None, device="cuda", batch_size=100, desc=None, dtype=torch.float32
):
    """
    From a DataSet of functions, discretise by evaluating all fields on the quadrature rule `quad`
    or its tensor product in 2D. The resulting data set can be used to train a model.
    """
    attributes = _make_attributes(dataset.data, quad_x, quad_y, device, torch.float64)
    discretised_data = None
    for indices in BatchIterator(len(dataset), batch_size, desc):
        batch_fun = dataset.get_batch(indices, device)
        batch = _discretise_batch(batch_fun, attributes, dtype)
        batch = recursive_to(batch, "cpu")

        # The first batch tells us which keys the dictionary will contain
        # After that, we collect for each key a list of values per batch
        if discretised_data is None:
            discretised_data = {key: [value] for (key, value) in batch.items()}
        else:
            for key, value in batch.items():
                discretised_data[key].append(value)

    discretised_data = {key: torch.cat(value) for (key, value) in discretised_data.items()}

    # Add the attributes
    return DataSet(
        discretised_data,
        recursive_to(attributes, dtype),
        dataset.config,
        dataset.filename,
        dataset.part,
    )


def discretise_and_assemble(
    dataset,
    model,
    quad_x,
    quad_y=None,
    device="cuda",
    batch_size=100,
    desc=None,
    dtype=torch.float32,
):
    """
    Given a dataset of functions, and a model, discretise the data set with the given
    quadrature rule, and assemble the inputs to the models branch network. This function
    does the same as calling discretise() and then assemble(), but by doing both steps
    batch by batch reduces memory usage if very fine quadrature rules are used, since the
    intermediate (discretised, but not assembled) data does not have to be stored all at
    once, only one batch at a time.
    """
    model.to(device)
    model.eval()
    attributes = _make_attributes(dataset.data, quad_x, quad_y, device, torch.float64)
    attributes_dtype = recursive_to(attributes, dtype)
    branch_data = None
    for indices in BatchIterator(len(dataset), batch_size, desc):
        batch_fun = dataset.get_batch(indices, device)
        batch = _discretise_batch(batch_fun, attributes, dtype)
        batch.update(attributes_dtype)
        batch["problem"] = batch_fun["problem"]
        with torch.no_grad():
            branch_batch = model.preassemble_batch(batch)

        branch_batch = recursive_to(branch_batch, "cpu")
        branch_batch = recursive_to(branch_batch, dtype)
        # The first batch tells us which keys the dictionary will contain
        # After that, we collect for each key a list of values per batch
        if branch_data is None:
            branch_data = {key: [value] for (key, value) in branch_batch.items()}
        else:
            for key, value in branch_batch.items():
                branch_data[key].append(value)

    branch_data = {key: torch.cat(value) for (key, value) in branch_data.items()}

    # Add the attributes
    branch_attributes = model.preassemble_attributes(attributes_dtype)
    branch_attributes = recursive_to(branch_attributes, "cpu")
    branch_attributes = recursive_to(branch_attributes, dtype)
    model.to("cpu")
    return DataSet(branch_data, branch_attributes, dataset.config, dataset.filename, dataset.part)


def assemble(
    dataset,
    model,
    device="cuda",
    batch_size=100,
    desc=None,
):
    """
    Given a discretised data set and a model, assemble a new data set containing the inputs to the
    model's branch net. This can then be used to train only the branch net.
    """
    branch_data = None
    model.to(device)
    # For each batch in the data, call the implementation that produces the branch-only data
    for indices in BatchIterator(len(dataset), batch_size, desc):
        batch = dataset.get_batch(indices, device)
        with torch.no_grad():
            branch_batch = model.preassemble_batch(batch)

        branch_batch = recursive_to(branch_batch, "cpu")
        # The first batch tells us which keys the dictionary will contain
        # After that, we collect for each key a list of values per batch
        if branch_data is None:
            branch_data = {key: [value] for (key, value) in branch_batch.items()}
        else:
            for key, value in branch_batch.items():
                branch_data[key].append(value)

    branch_data = {key: torch.cat(value) for (key, value) in branch_data.items()}

    model.to("cpu")
    # Add the attributes
    branch_attributes = model.preassemble_attributes(dataset.attributes)
    return DataSet(branch_data, branch_attributes, dataset.config, dataset.filename, dataset.part)


def discretise_and_test(
    dataset: DataSet,
    model,
    *,
    quad_x,
    quad_y=None,
    device,
    batch_size,
    desc=None,
    dtype=torch.float32,
):
    """
    Evaluates the given model on the data set, returning the relative L² error per problem. The
    dataset is discretised with the given quadrature rule(s).
    """
    errs = []
    model.to(device)
    # model.to(dtype)
    model.eval()
    attributes = _make_attributes(dataset.data, quad_x, quad_y, device, dtype=torch.float64)
    attributes = recursive_to(attributes, dtype)
    for indices in BatchIterator(len(dataset), batch_size, desc):
        batch_fun = dataset.get_batch(indices, device)
        batch = _discretise_batch(batch_fun, attributes, dtype)
        batch.update(attributes)
        batch["problem"] = batch_fun["problem"]

        with torch.no_grad():
            u_pred = model(batch)

        u_actual = batch["u"]
        err = relative_errors(
            u_pred, u_actual, attributes["u/weights_x"], attributes["u/weights_y"]
        ).sqrt()
        errs.append(err)

    errs = torch.cat(errs)
    return errs
