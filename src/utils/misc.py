import os
import pathlib
from typing import Any, Union
import torch
import numpy as np
from .quadrature import TensorGrid
from tqdm import tqdm


def to_complex(t):
    """
    Given a torch.Tensor t, promote t from a real tensor to a complex tensor with zero imaginary
    part. If t is already complex, then the original tensor is returned.
    """
    if torch.is_complex(t):
        return t
    else:
        return t.type(t.dtype.to_complex())


def einsum(formula, *args):
    """
    This is a wrapper around torch.einsum that handles the case that some of the input tensors are
    real and others complex. NumPy handles this case as expected, but PyTorch raises an error when
    real and complex types are mixed. This function automatically promotes all tensors to complex if
    any are already complex. Note that it does not handle any other conversions, such as between
    single and double precision, or between devices.
    """
    any_complex = any(torch.is_complex(t) for t in args)
    if any_complex:
        args = tuple(to_complex(t) for t in args)

    return torch.einsum(formula, *args)


class BatchIterator:
    def __init__(self, N, B, desc=None):
        """
        Create an iterator that loops over fixed-length subranges of `range(0, N)`. For example,
        `BatchIterator(10, 3)` yields parts of `range(0, 10)` so `range(0, 3)`, `range(3, 6)`,
        `range(6, 9)`, and finally `range(9, 10)`.

        If `desc` is given, the iterator also shows a progress bar (using tqdm).
        """
        self.N = N
        self.B = B
        self.iter = (
            None if desc is None else tqdm(total=N, desc=desc, ascii=True, dynamic_ncols=True)
        )

    def __iter__(self):
        for i0 in range(0, self.N, self.B):
            indices = range(i0, min(i0 + self.B, self.N))
            yield indices
            if self.iter is not None:
                self.iter.update(len(indices))

        if self.iter is not None:
            self.iter.close()


def save(object, location, allow_exists=False):
    """
        save(object, location, allow_exists=False)

    Very similar to `torch.save(object, location)`, but with different behaviour depending on
    existence of the save location.
    - if `allow_exists=False` (default), then a `FileExistsError` is raised if the specified save
      location already exists.
    - if `allow_exists=True`, then if a file already exists at the specified location, it is
      overwritten. This is also the behaviour of `torch.save`.
    - Finally, regardless of the value of `allow_exists`, this function will create all
      subdirectories necessary for the location to exist.
    """
    location = pathlib.Path(location)
    directory = location.parent
    file_exists = location.exists()

    if file_exists and not allow_exists:
        raise FileExistsError(location, "File exists")

    try:
        os.makedirs(directory)
    except FileExistsError:
        # The directories are allowed to exist, so this error can be ignored
        pass
    torch.save(object, location)


def nr_params(model: torch.nn.Module) -> int:
    """
    Returns the total parameter count of a PyTorch model
    """
    return sum(t.numel() for t in model.parameters())


def _recursive_to(obj: Any, d, strict=True) -> Any:
    """
    Given a dictionary containing only tensors or other dictionaries, recursively move all the
    tensors onto the specified device or dtype. This function (currently) accepts torch.Tensors,
    torch.nn.Modules, TensorGrids, and dictionaries (which are handled recursively). Values of
    other types result in an error (if strict==True, which is the default), or are left unchanged
    (if strict==False).

    `d` should be anything that can be used with the `.to()` method of torch tensors and modules:
    - Numeric types such as `torch.float32` and `torch.float64`
    - Devices such as "cpu" or `torch.device("cuda:0")`

    If strict==True, which is the default, objects that cannot be converted yield an error. If
    strict==False, these objects are returned unchanged
    """
    if isinstance(obj, torch.nn.Module) or isinstance(obj, TensorGrid):
        return obj.to(d)
    elif isinstance(obj, torch.Tensor):
        # Make sure that complex tensors are cast to the correct complex type, e.g. complexf64
        # instead of complex32
        if d in [torch.float16, torch.float32, torch.float64] and torch.is_complex(obj):
            return obj.to(d.to_complex())
        else:
            # We are moving to a device, or the tensor is already of a real type
            # In either case, we can call .to() directly
            return obj.to(d)

    elif isinstance(obj, dict):
        return {k: recursive_to(v, d, strict) for (k, v) in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(_recursive_to(t, d) for t in obj)
    elif strict:
        raise ValueError(
            f"recursive_to only works on Tensors, Modules, and dictionaries (got type {type(obj)})"
        )
    else:
        return obj


def recursive_to(d: dict[str, Any], device, strict=False) -> dict[str, Any]:
    """
    Given a dictionary containing only tensors or other dictionaries, recursively move all the
    tensors onto the specified device or dtype. This function (currently) accepts torch.Tensors,
    torch.nn.Modules, TensorGrids, and dictionaries (which are handled recursively). Values of other
    types result in an error (if strict==True), or are left unchanged (if strict==False).

    `d` should be anything that can be used with the `.to()` method of torch tensors and modules:
    - Numeric types such as `torch.float32` and `torch.float64`
    - Devices such as "cpu" or `torch.device("cuda:0")`

    If strict==True, which is the default, objects that cannot be converted yield an error. If
    strict==False, these objects are returned unchanged
    """
    return {k: _recursive_to(v, device, strict) for (k, v) in d.items()}


def make_tree(obj):
    int_types = [int, np.int64, torch.int64]
    float_types = [float, np.float64, torch.float32, torch.float64]
    if type(obj) in int_types:
        yield f"{int(obj)} ({type(obj).__name__})"
    elif type(obj) in float_types:
        yield f"{float(obj)} ({type(obj).__name__})"
    elif isinstance(obj, dict):
        if len(obj) == 0:
            yield "(empty dict)"
            return
        yield f"dict with {len(obj)} items:"
        for key, value in obj.items():
            value_desc = make_tree(value)
            is_first = True
            for line in value_desc:
                if is_first:
                    yield f"- {key}: {line}"
                    is_first = False
                else:
                    yield f"  {line}"
    elif isinstance(obj, torch.Tensor):
        shape = " × ".join(str(s) for s in obj.shape)
        yield f"tensor, {shape} of {obj.dtype}"
    elif isinstance(obj, np.ndarray):
        shape = " × ".join(str(s) for s in obj.shape)
        yield f"np array, {shape} of {obj.dtype}"
    else:
        yield f"{type(obj)}"


def print_tree(obj):
    for line in make_tree(obj):
        print(line)


def tree(d, indent=0):
    """
    Given a dictionary, print a summary of the contents
    """
    if indent == 0:
        print("contents:", flush=False)

    for key, value in d.items():
        print("  " * indent, "- ", key, ": ", sep="", end="", flush=False)
        if isinstance(value, dict):
            print(flush=False)
            tree(value, indent + 1)
        elif isinstance(value, torch.Tensor):
            print("tensor, ", "×".join(map(str, value.shape)), sep="", flush=False)
        elif isinstance(value, np.ndarray):
            print("np array, ", "×".join(map(str, value.shape)), sep="", flush=False)
        else:
            print(type(value), sep="", flush=False)

    if indent == 0:
        print()


def roll_avg(arr: Union[torch.Tensor, list], k) -> torch.Tensor:
    """
    Given a torch.tensor and an integer k >= 1, compute the rolling averages of the array. In
    words, the output will be an array of length len(input) + 1 - k, where output[i] is the average
    of input[i:i+k].
    """
    arr = torch.tensor(arr) if isinstance(arr, list) else arr
    newlen = arr.shape[0] + 1 - k
    result = torch.zeros_like(arr[:newlen])
    for i in range(k):
        result += arr[i : i + newlen]

    return result / k


def _make_param_count_tree(model):
    result = {
        "children": {},
        "parameters": {},
        "total": 0,
        "type": type(model).__name__,
    }
    for name, param in model.named_parameters(recurse=False):
        sz = param.numel()
        description = "×".join(map(str, param.shape)) + " " + str(param.dtype)
        if param.is_complex():
            sz *= 2
        result["parameters"][name] = {"description": description, "count": sz}
        result["total"] += sz

    for name, child in model.named_children():
        result["children"][name] = _make_param_count_tree(child)
        result["total"] += result["children"][name]["total"]

    return result


def _show_param_count_tree(tree, indent=0, name=None):
    prefix = "  " * indent
    if name is None:
        print(f"{prefix}- {tree['type']}: {tree['total']}")
    else:
        print(f"{prefix}- {name}: {tree['total']}")
    for name, param in tree["parameters"].items():
        print(f"{prefix}  - {name}: {param['description']}")

    for name, child in tree["children"].items():
        if child["total"] != 0:
            # no need to print trees for modules with zero parameters
            _show_param_count_tree(child, indent + 1, name)


def print_param_count_tree(model):
    """
    Prints an overview showing how many parameters each part of the given model contains.
    """
    tree = _make_param_count_tree(model)
    _show_param_count_tree(tree)


def norm_sq(x, weights_x, weights_y, weights_z=None):
    """
    Returns the squared norm of the matrix `x`, assumed to be B×X×Y, weighted by the quadrature
    weights weights_x and weights_y. If `weights_z` is given, `x` is instead assumed to be B×X×Y×Z.
    """
    # Assume tensorised
    if weights_z is None:
        return torch.einsum("Bxy,x,y->B", x.abs().square(), weights_x, weights_y)
    else:
        return torch.einsum("Bxyz,x,y,z->B", x.abs().square(), weights_x, weights_y, weights_z)


def relative_errors(
    predicted: torch.Tensor,
    actual: torch.Tensor,
    weights_x: torch.Tensor,
    weights_y: torch.Tensor,
    weights_z: Union[torch.Tensor, None] = None,
) -> torch.Tensor:
    if predicted.shape != actual.shape:
        raise RuntimeError(
            f"relative_error: got differing shapes {predicted.shape=}, {actual.shape=}"
        )

    errs = norm_sq(predicted - actual, weights_x, weights_y, weights_z)
    norms = norm_sq(actual, weights_x, weights_y, weights_z)
    return errs / norms


def rms(t: torch.Tensor) -> torch.Tensor:
    return t.square().mean().sqrt()


def relerr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise RuntimeError(
            f"relerr(a, b) expects tensors of the same shape, got {a.shape=}, {b.shape=}"
        )
    return rms(a - b) / rms(b)
