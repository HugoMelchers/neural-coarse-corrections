import torch

from src import *

device = "cuda:0"
torch.set_num_threads(2)
with open(__file__) as f:
    script = f.read()

branch = nn.Sequential(
    nn.Linear(336, 60),
    nn.LeakyReLU(),
    nn.Linear(60, 60),
    nn.LeakyReLU(),
    nn.Linear(60, 144),
)

basis = TensorBasis(BSpline.uniform((0, 1), 9, 3))
config = {"sga": "bspline", "scale": True}
outputs = "values"
model = RINO("diffusion", branch, basis, config, outputs)

data = DataSet.load("diffusion-dataset-3")
quad = "gl9x7"
data_da = discretise(data, quad, desc="Discretising", device=device)
train, validate, test = data_da.split(8000, 1000, 1000)

train_model(
    model=model,
    kind="model",
    data_train=train,
    data_validate=validate,
    nr_epochs=20000,
    output_name="diffusion-dataset-3/rino",
    attributes={"script": script},
    device=device,
)
