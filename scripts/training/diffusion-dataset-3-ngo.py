import torch

from src import *

device = "cuda:0"
torch.set_num_threads(2)
with open(__file__) as f:
    script = f.read()


basis = TensorBasis(BSpline.uniform((0, 1), 9, 3))
system_net = nn.Sequential(
    UNet.make(
        in_channels=1,
        out_channels=24,
        channels=[24, 24, 24, 24],
        lifting_dim=24,
        projection_dim=None,
        activation="leakyrelu",
        widths=[2, 2, 3],
        strides=[2, 2, 3],
        pointwise_dim=None,
        input_size=[12, 12],
        skip_type="add",
        bias=True,
    ),
    nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=3, stride=3),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(in_channels=24, out_channels=24, kernel_size=2, stride=2),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(in_channels=24, out_channels=1, kernel_size=2, stride=2),
    StrictReshapeLayer((1, 144, 144), (144, 144)),
)
config = {"sga": "bspline", "scale": True}
outputs = "values"
model = NGO(problem="diffusion", system_net=system_net, basis=basis, config=config, outputs=outputs)

data = DataSet.load("diffusion-dataset-3")
quad = "gl9x7"
data_da = discretise_and_assemble(data, model, quad, desc="Discretising", device=device)
train, validate, test = data_da.split(8000, 1000, 1000)

train_model(
    model=model,
    kind="branch",
    data_train=train,
    data_validate=validate,
    nr_epochs=20000,
    output_name="diffusion-dataset-3/ngo",
    attributes={"script": script},
    device=device,
)
