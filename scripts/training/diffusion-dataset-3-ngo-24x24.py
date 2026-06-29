import torch

from src import *

device = "cuda:0"
torch.set_num_threads(2)
with open(__file__) as f:
    script = f.read()


C = 20
basis = TensorBasis(BSpline.uniform((0, 1), 21, 3))
system_net = nn.Sequential(
    UNet.make(
        in_channels=1,
        out_channels=C,
        channels=[C, C, C, C],
        lifting_dim=C,
        projection_dim=None,
        activation="leakyrelu",
        widths=[2, 3, 4],
        strides=[2, 3, 4],
        pointwise_dim=None,
        input_size=[24, 24],
        skip_type="add",
        bias=True,
    ),
    nn.ConvTranspose2d(in_channels=C, out_channels=C, kernel_size=4, stride=4),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(in_channels=C, out_channels=C, kernel_size=3, stride=3),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(in_channels=C, out_channels=1, kernel_size=2, stride=2),
    StrictReshapeLayer((1, 576, 576), (576, 576)),
)
config = {"sga": "bspline", "scale": True}
outputs = "values"
model = NGO(problem="diffusion", system_net=system_net, basis=basis, config=config, outputs=outputs)

data = DataSet.load("diffusion-dataset-3-24x24")
quad = "gl21x7"
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
