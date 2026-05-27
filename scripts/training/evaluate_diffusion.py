from src.training.testing import test_models

quad_deeponet = ("t12->gl9x7", {"u": "gl9x7", "default": "t12"})

# fmt: off
datasets = {
    "train": "diffusion-dataset-1:0-8000",
    "validate": "diffusion-dataset-1:8000-9000",
    "test": "diffusion-dataset-1:9000-10000",
}

data = [
    {"name": "DeepONet"  , "path": "diffusion-dataset-1/deeponet", "quad": quad_deeponet},
    {"name": "RINO"      , "path": "diffusion-dataset-1/rino"    , "quad": "gl9x7"      },
    {"name": "VarMiON"   , "path": "diffusion-dataset-1/varmion" , "quad": quad_deeponet},
    {"name": "NGO"       , "path": "diffusion-dataset-1/ngo"     , "quad": "gl9x7"      },
    {"name": "Projection", "path": "projection"                  , "quad": "gl9x7"      },
]
# fmt: on

for entry in data:
    quad = entry["quad"]
    if isinstance(quad, str):
        # quad is a string identifying a quadrature rule, like "m100" or "gl9x7"
        for name, filename in datasets.items():
            entry["data/" + name] = (filename, quad)
    else:
        # quad is a tuple consisting of a name (like t12->gl9x7) and a dict containing the
        # discretisation rules
        for name, filename in datasets.items():
            entry["data/" + name] = (filename, quad[0], quad[1])

test_models(data, table_fmt="markdown")


# fmt: off
datasets = {
    "train": "diffusion-dataset-2:0-8000",
    "validate": "diffusion-dataset-2:8000-9000",
    "test": "diffusion-dataset-2:9000-10000",
}

data = [
    {"name": "DeepONet"  , "path": "diffusion-dataset-2/deeponet", "quad": quad_deeponet},
    {"name": "RINO"      , "path": "diffusion-dataset-2/rino"    , "quad": "gl9x7"      },
    {"name": "VarMiON"   , "path": "diffusion-dataset-2/varmion" , "quad": quad_deeponet},
    {"name": "NGO"       , "path": "diffusion-dataset-2/ngo"     , "quad": "gl9x7"      },
    {"name": "Projection", "path": "projection"                  , "quad": "gl9x7"      },
]
# fmt: on

for entry in data:
    quad = entry["quad"]
    if isinstance(quad, str):
        # quad is a string identifying a quadrature rule, like "m100" or "gl9x7"
        for name, filename in datasets.items():
            entry["data/" + name] = (filename, quad)
    else:
        # quad is a tuple consisting of a name (like t12->gl9x7) and a dict containing the
        # discretisation rules
        for name, filename in datasets.items():
            entry["data/" + name] = (filename, quad[0], quad[1])

test_models(data, table_fmt="markdown")


# fmt: off
datasets = {
    "train": "diffusion-dataset-3:0-8000",
    "validate": "diffusion-dataset-3:8000-9000",
    "test": "diffusion-dataset-3:9000-10000",
}

data = [
    {"name": "DeepONet"  , "path": "diffusion-dataset-3/deeponet", "quad": quad_deeponet},
    {"name": "RINO"      , "path": "diffusion-dataset-3/rino"    , "quad": "gl9x7"      },
    {"name": "VarMiON"   , "path": "diffusion-dataset-3/varmion" , "quad": quad_deeponet},
    {"name": "NGO"       , "path": "diffusion-dataset-3/ngo"     , "quad": "gl9x7"      },
    {"name": "Projection", "path": "projection"                  , "quad": "gl9x7"      },
]
# fmt: on

for entry in data:
    quad = entry["quad"]
    if isinstance(quad, str):
        # quad is a string identifying a quadrature rule, like "m100" or "gl9x7"
        for name, filename in datasets.items():
            entry["data/" + name] = (filename, quad)
    else:
        # quad is a tuple consisting of a name (like t12->gl9x7) and a dict containing the
        # discretisation rules
        for name, filename in datasets.items():
            entry["data/" + name] = (filename, quad[0], quad[1])

test_models(data, table_fmt="markdown")
