from src.training.testing import test_models

# fmt: off
datasets = {
    "train": "helmholtz:0-8000",
    "validate": "helmholtz:8000-9000",
    "test": "helmholtz:9000-10000",
}

quad_deeponet = ("t12->gl9x7", {"u": "gl9x7", "default": "t12"})
data = [
    {"name": "DeepONet"  , "path": "helmholtz/deeponet", "quad": quad_deeponet},
    {"name": "RINO"      , "path": "helmholtz/rino"    , "quad": "gl9x7"      },
    {"name": "VarMiON"   , "path": "helmholtz/varmion" , "quad": quad_deeponet},
    {"name": "NGO"       , "path": "helmholtz/ngo"     , "quad": "gl9x7"      },
    {"name": "Projection", "path": "projection"        , "quad": "gl9x7"      },
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
