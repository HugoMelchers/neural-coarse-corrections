# Support for the 3D diffusion problem. The preconditioning functions are not imported here (as in
# 2D), since they depend on additional packages; use `src.diffusion3d.preconditioning` directly.
from .basis import TensorGrid3D, TensorBasis3D, Functions3D
from .discretise import discretise_3d, discretise_and_assemble_3d, discretise_and_test_3d
from .ngo2 import NGO_3D, data_ngo_assemble_diffusion_3d
from .assemble import assemble_diffusion_3d

__all__ = [
    "TensorGrid3D",
    "TensorBasis3D",
    "Functions3D",
    "discretise_3d",
    "discretise_and_assemble_3d",
    "discretise_and_test_3d",
    "NGO_3D",
    "data_ngo_assemble_diffusion_3d",
    "assemble_diffusion_3d",
]
