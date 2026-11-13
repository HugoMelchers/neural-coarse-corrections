# An implementation of U-Nets (https://arxiv.org/abs/1505.04597). In a U-Net, an input image (or
# matrix) is repeatedly down-scaled by convolution and pooling layers, followed by a small dense
# network (or a single layer) and then (transpose) convolutions that inflate the image again. The
# important extra ingredient is that there are skip connections between different intermediate
# results with the same image size.

import torch
from torch import nn
from .misc import ReshapeLayer, StrictReshapeLayer, get_activation
from typing import Union, List, Tuple

from warnings import warn


class UNet(nn.Module):
    config = {}

    def __init__(
        self,
        *,
        lifting,
        projection,
        convs,
        tr_convs,
        skips,
        dense,
        dense_skip=nn.Identity(),
        skip_type="add",
        config={},
    ):
        """
        Construct a UNet with `n = len(convs)` layers.
        - `lifting`: a network that will be applied pointwise to each point in the input matrices
          (i.e. if the input is B × c × m × n), then the lifting acts on the 'c' dimension.
        - `projection`: a network that will be applied pointwise to each point in the output
          matrices.
        - `convs` should be a list of `n` `nn.Module`s. They don't have to be pure convolutions,
          since they can also be `Sequential`s containing activation functions.
        - `tr_convs` should be a list of `n` `nn.Module`s. These should expand the array size again.
          In terms of input and output sizes, the first tr_conv should be the inverse of the first
          conv, the second tr_conv should be the inverse of the second conv, and so on.
        - `dense`: should be a dense network that maps the smallest array to something of the same
          size.
        - `dense_skip`: same as `dense`, but can also be `True` to represent `nn.Identity()` or
          `False` to disable the last skip connection
        - `skips` should be a list of `n` booleans or `nn.Module`s that act pointwise. These are
          usually 1x1 convolutions followed by activation functions.
        - `skip_type` should be either "add" or "concat". "add" means that the skip modules are
          applied to the stored arrays, and then added pointwise to the output of the transpose
          convolutions. "concat" means that the transpose convolution output and the stored arrays
          are concatenated and then passed through the skip connection
        """
        super().__init__()
        self.lifting = lifting
        self.projection = projection
        self.convs = nn.ModuleList(convs)
        self.dense = dense
        self.dense_skip = dense_skip
        self.tr_convs = nn.ModuleList(tr_convs)
        self.skips = nn.ModuleList(skips)
        self.skip_type = skip_type
        self.config = config
        fix_adjoint = self.config.get("fix-adjoint", False)
        if not fix_adjoint:
            warn(
                'UNet construct with fix_adjoint set to False. Add `"fix-adjoint": True` to the '
                "config to ensure adjoints of the linear UNet are correct",
                stacklevel=2,
            )

    def forward(self, img, callback=None, return_all=False):
        fix_adjoint = self.config.get("fix-adjoint", False)
        # If an input size is given, check that this input has the right size
        if (input_size := self.config.get("input-size")) is not None:
            err = f"UNet: expected shape (B, {', '.join(map(str, input_size))}), got {img.shape}"
            assert img.ndim == 1 + len(input_size), err
            assert tuple(img.shape[1:]) == tuple(input_size), err

        # Step 1: lifting
        if self.lifting is not None:
            img = self.lifting(img)

        # Step 2: apply the convolutions, downsampling the image, storing the intermediate results
        store = [img]
        for conv in self.convs:
            img = conv(img)
            store.append(img)

        # Step 3: apply the dense network
        img = self.dense(img) + self.dense_skip(img)

        # Step 4: apply the transpose convolutions, adding the skip connection term after each layer
        # This happens in reverse order, so from the dense output to larger and larger array sizes
        for i in reversed(range(len(self.tr_convs))):
            if self.skip_type == "concat":
                img = self.tr_convs[i](img)
                img = torch.cat([img, store[i]], dim=1)
                img = self.skips[i](img)
            elif self.skip_type == "add":
                img = self.tr_convs[i](img) + self.skips[i](store[i])

        # Step 5: projection
        if self.projection is not None:
            img = self.projection(img)

        return img

    @staticmethod
    def make(
        *,
        in_channels: int,
        out_channels: int,
        channels: Union[List[int], int],
        lifting_dim: int,
        projection_dim: int,
        activation: Union[str, nn.Module],
        widths: List[int],
        strides: List[int],
        pointwise_dim: int,
        input_size: Union[int, Tuple[int, ...]],
        skip_type: str,
        bias: bool = True,
        complex: bool = False,
        dim: int = 2,
    ):
        """
        Construct a UNet. Parameters:
        - `in_channels: int`: the number of channels of the input
        - `out_channels: int`: the number of channels of the output
        - `channels: Union[List[int], int]`: the number of channels of the hidden layers. Either a
          list of ints, or a single int to use the same number of channels for all internal layers
        - `lifting_dim: int`: the dimension of the pointwise lifting. Set to `None` to disable the
          lifting layer
        - `projection_dim: int`: the dimension of the pointwise projection. Set to `None` to disable
          the projection layer
        - `activation: Union[str, nn.Module]`: either a PyTorch activation function, or a string
          like "relu" or "identity"
        - `widths: List[int]`: the widths of the convolution and transpose convolution kernels
        - `strides: List[int]`: the strides of the convolution and transpose convolution kernels
        - `pointwise_dim: int`: the number of channels for the pointwise mapping layers. Is ignored
          when `skip_type = "add"`. Can also be set to None, to use only a single linear layer (from
          2C to C channels) in the pointwise mapping, instead of 2 (from 2C to `pointwise_dim` and
          then to C channels)
        - `input_size: Union[int, Tuple[int, ...]]`: the dimensions of the input array
        - `skip_type: str`
        - `bias: bool = True`
        - `complex: bool = False`
        - `dim: int = 2`: the number of spatial dimensions of the input, i.e. 2 for images of shape
          B × C × nx × ny (using 2D convolutions) or 3 for volumes of shape B × C × nx × ny × nz
          (using 3D convolutions)
        """
        if isinstance(activation, str):
            activation = get_activation(activation)
        else:
            assert isinstance(activation, nn.Module)

        dtype = torch.complex64 if complex else torch.float32
        assert dim in (2, 3), f"UNet: dim must be 2 or 3, got {dim}"
        Conv = nn.Conv2d if dim == 2 else nn.Conv3d
        ConvTranspose = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d

        # If our UNet is affine or linear, we can remove some layers since the concatenation of
        # affine layers is again affine, and the concatenation of linear layers is again linear
        is_affine = isinstance(activation, nn.Identity)
        is_linear = is_affine and (not bias)

        if isinstance(input_size, int):
            input_size = (input_size,) * dim
        assert len(input_size) == dim, f"UNet: expected {dim}-dimensional input_size"

        n_convs = len(widths)
        assert len(strides) == n_convs
        if isinstance(channels, int):
            channels = [channels] * (n_convs + 1)
        else:
            assert isinstance(channels, list)
            assert len(channels) == n_convs + 1

        if lifting_dim is None:
            lifting = None
        elif is_affine:
            lifting = nn.Sequential(
                Conv(in_channels, channels[0], kernel_size=1, stride=1, bias=bias, dtype=dtype),
            )
        else:
            lifting = nn.Sequential(
                Conv(in_channels, lifting_dim, kernel_size=1, stride=1, bias=bias, dtype=dtype),
                activation,
                Conv(lifting_dim, channels[0], kernel_size=1, stride=1, bias=bias, dtype=dtype),
            )

        # Size of the array after all the strided convolutions
        sizes = list(input_size)
        for stride in strides:
            sizes = [n // stride for n in sizes]

        convs = [
            nn.Sequential(
                Conv(channels[i], channels[i + 1], widths[i], strides[i], bias=bias, dtype=dtype),
                activation,
            )
            for i in range(n_convs)
        ]

        dense_size = channels[-1]
        for n in sizes:
            dense_size *= n
        dense = nn.Sequential(
            StrictReshapeLayer((channels[-1], *sizes), (dense_size,)),
            nn.Linear(dense_size, dense_size, bias=bias, dtype=dtype),
            activation,
            StrictReshapeLayer((dense_size,), (channels[-1], *sizes)),
        )

        tr_convs = [
            nn.Sequential(
                ConvTranspose(
                    channels[i + 1], channels[i], widths[i], strides[i], bias=bias, dtype=dtype
                ),
                activation,
            )
            for i in range(n_convs)
        ]

        # The skip connections should now be pointwise dense NNs
        if skip_type == "concat":
            if pointwise_dim is None:
                skips = [
                    nn.Sequential(
                        Conv(2 * channels[i], channels[i], 1, 1, bias=bias, dtype=dtype),
                        activation,
                    )
                    for i in range(n_convs)
                ]
            elif is_affine:
                skips = [
                    Conv(2 * channels[i], channels[i], 1, 1, bias=bias, dtype=dtype)
                    for i in range(n_convs)
                ]
            else:
                skips = [
                    nn.Sequential(
                        Conv(2 * channels[i], pointwise_dim, 1, 1, bias=bias, dtype=dtype),
                        activation,
                        Conv(pointwise_dim, channels[i], 1, 1, bias=bias, dtype=dtype),
                        activation,
                    )
                    for i in range(n_convs)
                ]
        elif skip_type == "add":
            skips = [
                Conv(channels[i], channels[i], 1, 1, bias=bias, dtype=dtype) for i in range(n_convs)
            ]

        if projection_dim is None:
            projection = None
        elif is_affine:
            projection = Conv(
                channels[0], out_channels, kernel_size=1, stride=1, bias=bias, dtype=dtype
            )
        else:
            projection = nn.Sequential(
                Conv(channels[0], projection_dim, kernel_size=1, stride=1, bias=bias, dtype=dtype),
                activation,
                Conv(projection_dim, out_channels, kernel_size=1, stride=1, bias=bias, dtype=dtype),
            )

        unet = UNet(
            lifting=lifting,
            projection=projection,
            convs=convs,
            tr_convs=tr_convs,
            skips=skips,
            skip_type=skip_type,
            dense=dense,
            config={"input-size": (in_channels, *input_size), "fix-adjoint": True},
        )
        return unet
