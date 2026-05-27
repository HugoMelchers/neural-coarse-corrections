# Create a "Projection" output folder, which is not really a neural operator but a place for reference data
# - As a neural operator, it just returns the projection of the true solution onto the basis, acting as a baseline
# - As a preconditioner, it performs exact coarse solves
import torch

from src import *

basis = TensorBasis(BSpline.uniform((0, 1), 9, 3))
proj = ProjectOnBasis(basis)
save(proj, "output/projection/model.pt")
with open(__file__) as f:
    script = f.read()

with open("output/projection/script.py", "w") as f:
    f.write(script)
