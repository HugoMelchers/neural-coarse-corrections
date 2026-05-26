# This file contains the different assembly routines for NGOs. Since they depend on the PDE and
# are also different for data NGOs than for model NGOs, this produces a large number of functions
# which are therefore not included in ngo.py to keep the model architecture separate from the
# implementation details.

from .ngo_diffusion import data_ngo_assemble_diffusion
from .ngo_advection_diffusion import data_ngo_assemble_advdiff
from .ngo_helmholtz import data_ngo_assemble_helmholtz
