from src.training.testing import test_models

quad_deeponet = ("t12->gl9x7", {"u": "gl9x7", "default": "t12"})

# fmt: off
datasets = {
    "train": "advdiff-low-peclet:0-8000",
    "validate": "advdiff-low-peclet:8000-9000",
    "test": "advdiff-low-peclet:9000-10000",
}

data = [
    {"name": "DeepONet"  , "path": "advdiff/deeponet", "quad": quad_deeponet},
    {"name": "RINO"      , "path": "advdiff/rino"    , "quad": "gl9x7"      },
    {"name": "VarMiON"   , "path": "advdiff/varmion" , "quad": quad_deeponet},
    {"name": "NGO"       , "path": "advdiff/ngo"     , "quad": "gl9x7"      },
    {"name": "Projection", "path": "projection"      , "quad": "gl9x7"      },
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

test_models(data)
