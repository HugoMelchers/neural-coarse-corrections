import torch

from src import *

device = "cuda:0"
torch.set_num_threads(2)
with open(__file__) as f:
    script = f.read()


act = ComplexComponentWise(nn.LeakyReLU())
branch = nn.Sequential(
    ComplexComponentWise(nn.BatchNorm1d(336)),
    nn.Linear(336, 60, dtype=torch.complex64),
    act,
    nn.Linear(60, 60, dtype=torch.complex64),
    act,
    nn.Linear(60, 144, dtype=torch.complex64),
)

basis = TensorBasis(BSpline.uniform((0, 1), 9, 3))
config = {"sga": "bspline", "scale": True}
outputs = "values"
model = DeepONet("helmholtz", branch, basis, config, outputs)

data = DataSet.load("helmholtz")
quad = {"default": "t12", "u": "gl9x7"}
data_da = discretise(data, quad, desc="Discretising", device=device)
train, validate, test = data_da.split(8000, 1000, 1000)

train_model(
    model=model,
    kind="model",
    data_train=train,
    data_validate=validate,
    nr_epochs=20000,
    output_name="helmholtz/deeponet",
    attributes={"script": script},
    device=device,
)
