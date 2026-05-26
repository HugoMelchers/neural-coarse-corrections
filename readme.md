# Neural Operators for Preconditioning

NOTE: this repository is still under construction.

This project contains the code associated with the paper on neural operators as coarse-space corrections in two-level preconditioners: <http://arxiv.org/abs/2605.19867>. It is structured as follows:

- `src/` the folder containing the source code
  - `data_generation/`: code associated with generating training data
  - `training/`: code associated with neural operator architectures and training loops
    - `subnets/`: implementations of subnetworks (U-Nets etc.)
    - `models/`: implementations of neural operators
  - `preconditioning/`: code associated with using neural operators as preconditioners
  - `utils/`: other utilities
- `tests/`: contains tests for certain modules
- `datasets/`: stores generated datasets for training and testing networks
- `output/`: stores trained models as well as their errors etc.

# Installation

1. Download the source code: `git clone git@github.com:HugoMelchers/neural-coarse-corrections.git`. This library is known to work with Python 3.11. I recommend [PyEnv](https://github.com/pyenv/pyenv) to manage installed versions of Python.
2. Create a virtual environment and install the dependencies with `pip`:
    ```bash
    cd neural-coarse-corrections
    # Create virtual environment
    python -m venv venv
    # Enter the environment
    source ./venv/bin/activate
    # Make sure pip is installed and install requirements
    python -m ensurepip
    python -m pip install -r requirements
    ```

# Generating the training data

The dataset generation code has been written in Julia.

1. The Julia code has been written for Julia 1.11.7. I recommend using [Juliaup](https://github.com/JuliaLang/juliaup) to manage installed versions of Julia.
2. First enter the dataset generation code: `cd src/data_generation`
3. Launch Julia and install dependencies: `julia --project-. --threads=8`
    ```julia
    ]instantiate
    ```
    - If you get warnings about the manifest, or precompilation errors, check that you are using Julia 1.11.7
4. Make sure that the directory `datasets/` exists at the project root, _not_ under `src/data_generation/`. Else, exporting the data will fail. (Note: the Python code does create folders if they are missing, so this is not an issue for other steps)
5. Run `include("generate.jl")`. Now, the functions to generate the data sets are:
    - `generate_diffusion_dataset_1()`
    - `generate_diffusion_dataset_2()`
    - `generate_diffusion_dataset_3()`
    - `generate_advdiff()`
    - `generate_advdiff_low_peclet()`
    - `generate_advdiff_high_peclet()`
    - `generate_helmholtz()`

# Training the models

Use the scripts in the folder `scripts/training/` to train the models. For example, to train a RINO on the second diffusion data set, run

```bash
python -m scripts.training.diffusion-dataset-2-rino 
```

All these scripts are set to use the first available CUDA GPU. Each script has a line `device = "cuda:0"` that can be changed to run on another device. Each result is saved as a folder with contents

- `history.pt`: the training and validation error history during training
- `model.pt`: the trained model. This is a copy of the model that achieved the lowest validation error during training
- `model_before.pt`: a copy of the model with its original (randomly initialised weights)
- `model_description.pt`: a string describing the model
- `model_last.pt`: a copy of the model at the end of training. This is generally not the same as `model.pt` if the last training epoch didn't also yield the lowest validation error
- `optimiser.pt`: a copy of the optimiser after the final epoch. The data in `model_last.pt` and `optimiser.pt` could therefore be used together to resume training of the model.
- `script.py`: a copy of the exact script used to train the model

The results are stored in the `output/` folder, which will look as follows if all models are trained:

- `advdiff/`
    - `deeponet/`
    - `ngo/`
    - `rino/`
    - `varmion/`
- `diffusion-dataset-1/`
    - `deeponet/`
    - `ngo/`
    - `rino/`
    - `varmion/`
- `diffusion-dataset-2/`
    - `deeponet/`
    - `ngo/`
    - `rino/`
    - `varmion/`
- `diffusion-dataset-3/`
    - `deeponet/`
    - `ngo/`
    - `rino/`
    - `varmion/`
- `helmholtz/`
    - `deeponet/`
    - `ngo/`
    - `rino/`
    - `varmion/`

# Running the preconditioners

Use the scripts in the folder `scripts/preconditioning` to evaluate the trained neural operators as preconditioners.
