import torch

from src import *

device = "cpu"
torch.set_num_threads(2)
with open(__file__) as f:
    script = f.read()


# The data set uses 5 knot spans and cubic B-splines for theta and f, so use the same basis here,
# giving 8 basis functions per axis and N = 8³ = 512 basis functions in total
n = 8
N = n**3
C = 24
basis = TensorBasis3D(BSpline.uniform((0, 1), 5, 3))
# The system net maps the B × 1 × 8 × 8 × 8 input to N channels on each of the N voxels, which are
# then interpreted as the N × N matrix
system_net = nn.Sequential(
    UNet.make(
        in_channels=1,
        out_channels=N,
        channels=C,
        lifting_dim=C,
        projection_dim=C,
        activation="leakyrelu",
        widths=[2, 2],
        strides=[2, 2],
        pointwise_dim=None,
        input_size=n,
        skip_type="add",
        bias=True,
        dim=3,
    ),
    StrictReshapeLayer((N, n, n, n), (N, N)),
)
config = {"sga": "bspline", "scale": True}
outputs = "values"
model = NGO_3D(system_net=system_net, basis=basis, config=config, outputs=outputs)

print(nr_params(model))

data = DataSet.load("diffusion3d")
quad = "gl5x4"
data_da = discretise_and_assemble_3d(data, model, quad, desc="Discretising", device=device)
train, validate, test = data_da.split(8000, 1000, 1000)

train_model(
    model=model,
    kind="branch",
    data_train=train,
    data_validate=validate,
    nr_epochs=20000,
    output_name="diffusion3d/ngo",
    attributes={"script": script},
    device=device,
)
