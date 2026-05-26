import math

import torch
import h5py
import numpy as np

from .utils import recursive_to
from src.training.subnets.basis import RBF1D, BSpline, TensorBasis, Functions, TensorGrid, OneBasis


class DataSet:
    def __init__(self, data, attributes, config, filename, part):
        self.data = data
        self.attributes = attributes
        self.config = config
        self.len = len(data["u"])
        self.filename = filename
        self.part = part

    def get_slice(self, part):
        new_part = part if self.part is None else self.part[part]
        return DataSet(
            {key: value[part] for (key, value) in self.data.items()},
            self.attributes,
            self.config,
            self.filename,
            new_part,
        )

    def split(self, *sizes):
        """
        Split the data set up into multiple data sets with the given number of samples each. The sum
        of all the sizes should equal the size of the data set. Examples:
        - `train, test = data.split(9000, 1000)` splits a data set of 10000 samples up into 9000 for
          training and the remaining 1000 for testing
        - `train, validate, test = data.split(8000, 1000, 1000)` splits a data set of 10000 samples
          up into 8000 for training, 1000, for validation, and 1000 for testing
        - `train, _skip, validate, test = data.split(4000, -1, 1000, 1000)` splits a data set so
          that `train` contains the first 4000 samples, `validate` and `test` combined contain the
          last 2000 samples, and `_skip` contains the part in between `train` and `validate`. Only
          one `-1` can be given.
        """
        has_m1 = sizes.count(-1)
        assert has_m1 <= 1, f"data splits can contain at most one -1 (got {has_m1})"
        if has_m1 == 1:
            i = sizes.index(-1)
            sizes = list(sizes)
            # The sum of the sizes excluding the -1 should be at most the data length, so that we
            # get a non-negative size for the remainder. Note that instead of skipping the -1 in the
            # sum, I simply add 1
            allocated = 1 + sum(sizes)
            assert allocated <= len(
                self
            ), f"Cannot split data of length {len(self)} into sizes {tuple(sizes)}"
            sizes[i] = len(self) - allocated

        assert sum(sizes) == len(
            self
        ), f"Cannot split data of length {len(self)} into sizes {sizes}"
        start = 0
        results = []
        for size in sizes:
            stop = start + size
            results.append(self.get_slice(slice(start, stop)))
            start = stop

        return tuple(results)

    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.data}

    def get_batch(self, indices, device):
        batch = self[indices]
        batch.update(self.attributes)
        batch["problem"] = self.config["problem"]
        return recursive_to(batch, device)

    def to(self, device):
        self.data = recursive_to(self.data, device)
        self.attributes = recursive_to(self.attributes, device)
        return self

    def __len__(self):
        return self.len

    def identifier(self):
        if self.filename is None:
            return None
        elif self.part is None:
            return self.filename
        else:
            return f"{self.filename}:{self.part.start}-{self.part.stop}"

    @staticmethod
    def load(dataset_name):
        """
        Loads a dataset, identified by the given name and optional range.
        - `load("name")` will attempt to read the file `datasets/name.hdf5`
        - `load("name:700-800")` will read the file `datasets/name.hd5`, but only return samples
          700 through 800 (in range terminology, so equivalent to array[700:800])
        """
        if ":" in dataset_name:
            filename, part = dataset_name.split(":")
            start, stop = tuple(map(int, part.split("-")))
        else:
            filename = dataset_name
            start, stop = None, None

        file = h5py.File(f"datasets/{filename}.hdf5")
        config = dict(file.attrs)
        config = {k: v.decode("utf-8") if isinstance(v, bytes) else v for (k, v) in config.items()}
        attributes = {}
        data = {name: _get_field(file[name]) for name in file.keys()}
        dataset = DataSet(data, attributes, config, filename, None)

        if start is None:
            return dataset
        else:
            # Note: it is very important that we index with ranges instead of slices here. The main
            # difference between slice indexing and range indexing (for 1D indices at least) is that
            # indexing with a range creates a copy, whereas slicing creates a view of the underlying
            # data. That means that if we indexed using slices, the entire dataset would be kept
            # in memory, even if only a small part of it is ever accessible, leading to much higher
            # memory usage.
            return dataset.get_slice(range(start, stop))


def _get_basis(group):
    attrs = group.attrs
    basis_type = attrs["type"]
    if basis_type == b"rbf1d" or basis_type == "rbf1d":
        l = attrs["l"]
        nodes = torch.tensor(attrs["nodes"])
        assert nodes.ndim == 1, f"RBF1D: expected one-dimensional nodes, got shape {nodes.shape}"
        return RBF1D(nodes.shape[0], nodes, torch.full_like(nodes, math.log(l)))
    elif basis_type == b"bspline" or basis_type == "bspline":
        degree = attrs["degree"]
        knots = attrs["knots"]
        return BSpline.from_parts(degree, knots)
    elif basis_type == b"const" or basis_type == "const":
        return OneBasis()
    elif basis_type == b"tensor" or basis_type == "tensor":
        basis_x = _get_basis(group["x"])
        basis_y = _get_basis(group["y"])
        return TensorBasis(basis_x, basis_y)
    else:
        raise ValueError(f"unexpected basis type {basis_type}")


def _get_field(group):
    # In a HDF5 group the field coefficients are stored "directly"
    # The information on the basis is stored in the attributes
    basis = _get_basis(group["basis"])
    weights = torch.tensor(np.array(group["weights"]))
    if weights.ndim == 3:
        # Julia stores the coefficients in column-major ordering with axes ordered (x, y, number).
        # These are then loaded in row-major order with axes ordered (number, y, x). However, we
        # want them ordered (number, x, y), so we swap the last two axes. Note that for 1D functions
        # the Julia ordering is (x, number) which is then correctly loaded as (number, x), so
        # nothing needs to be done in that case.
        weights = weights.permute(0, 2, 1)

    return Functions(basis, weights)
