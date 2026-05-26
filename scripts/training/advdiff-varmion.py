import torch

from src import *

device = "cuda:0"
torch.set_num_threads(2)
with open(__file__) as f:
    script = f.read()


basis = TensorBasis(BSpline.uniform((0, 1), 9, 3))
linears = nn.Linear(192, 144, bias=False)
branch = nn.Sequential(
    StrictReshapeLayer((3 * 144,), (3, 12, 12)),
    UNet.make(
        in_channels=3,
        out_channels=12,
        channels=[12, 12, 12, 12],
        lifting_dim=12,
        projection_dim=None,
        activation="leakyrelu",
        widths=[2, 2, 3],
        strides=[2, 2, 3],
        pointwise_dim=None,
        input_size=[12, 12],
        skip_type="add",
        bias=True,
    ),
    nn.ConvTranspose2d(in_channels=12, out_channels=12, kernel_size=3, stride=3),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(in_channels=12, out_channels=12, kernel_size=2, stride=2),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(in_channels=12, out_channels=1, kernel_size=2, stride=2),
    StrictReshapeLayer((1, 144, 144), (144, 144)),
)
config = {"sga": "bspline", "scale": True}
outputs = "values"
model = VarMiON("advection-diffusion", linears, branch, basis, config, outputs)

data = DataSet.load("advdiff-low-peclet")
quad = {"default": "t12", "u": "gl9x7"}
data_da = discretise(data, quad, desc="Discretising", device=device)
train, validate, test = data_da.split(8000, 1000, 1000)

train_model(
    model=model,
    kind="model",
    data_train=train,
    data_validate=validate,
    nr_epochs=20000,
    output_name="advdiff/varmion",
    attributes={"script": script},
    device=device,
)
