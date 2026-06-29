# Create bar charts

from .tables import makeplot
from .styles import colors, alphas, styles

ns = [20, 40, 80, 160, 320]
sm = "G_diffusion_0.05_1_1.5"

# fmt: off
pth = f"precondition/diffusion-dataset-1-9000-10000"
kinds = ["noprec", "smooth", "exact", "deeponet", "rino", "varmion", "ngo"]
cfg = {
    "fileses": [
        ("None"         , [f"projection/{pth}/fgmres/n{n}/noprec"        for n in ns]),
        ("Smoother only", [f"projection/{pth}/fgmres/n{n}/{sm}"          for n in ns]),
        ("Exact"        , [f"projection/{pth}/fgmres/n{n}/A-DEF1-{sm}"   for n in ns]),
        ("DeepONet"     , [f"diffusion-dataset-1/deeponet/{pth}/fgmres/n{n}/A-DEF1-{sm}" for n in ns]),
        ("RINO"         , [f"diffusion-dataset-1/rino/{pth}/fgmres/n{n}/A-DEF1-{sm}"     for n in ns]),
        ("VarMiON"      , [f"diffusion-dataset-1/varmion/{pth}/fgmres/n{n}/A-DEF1-{sm}"  for n in ns]),
        ("NGO"          , [f"diffusion-dataset-1/ngo/{pth}/fgmres/n{n}/A-DEF1-{sm}"      for n in ns]),
    ],
    "colours": [colors[kind] for kind in kinds],
    "alphas": [alphas[kind] for kind in kinds],
    "seps": [3],
    "outfile": "diffusion/barchart-FGMRES-dataset-1",
    "solver": "F-GMRES",
    "title": "",
    "loc": "upper center",
}
makeplot(ns, **cfg)


pth = f"precondition/diffusion-dataset-2-9000-10000"
kinds = ["noprec", "smooth", "exact", "deeponet", "rino", "varmion", "ngo"]
cfg = {
    "fileses": [
        ("None"         , [f"projection/{pth}/fgmres/n{n}/noprec"        for n in ns]),
        ("Smoother only", [f"projection/{pth}/fgmres/n{n}/{sm}"          for n in ns]),
        ("Exact"        , [f"projection/{pth}/fgmres/n{n}/A-DEF1-{sm}"   for n in ns]),
        ("DeepONet"     , [f"diffusion-dataset-2/deeponet/{pth}/fgmres/n{n}/A-DEF1-{sm}" for n in ns]),
        ("RINO"         , [f"diffusion-dataset-2/rino/{pth}/fgmres/n{n}/A-DEF1-{sm}"     for n in ns]),
        ("VarMiON"      , [f"diffusion-dataset-2/varmion/{pth}/fgmres/n{n}/A-DEF1-{sm}"  for n in ns]),
        ("NGO"          , [f"diffusion-dataset-2/ngo/{pth}/fgmres/n{n}/A-DEF1-{sm}"      for n in ns]),
    ],
    "colours": [colors[kind] for kind in kinds],
    "alphas": [alphas[kind] for kind in kinds],
    "seps": [3],
    "outfile": "diffusion/barchart-FGMRES-dataset-2",
    "solver": "F-GMRES",
    "title": "",
}
makeplot(ns, **cfg)


pth = f"precondition/diffusion-dataset-3-9000-10000"
kinds = ["noprec", "smooth", "exact", "deeponet", "rino", "varmion", "ngo"]
cfg = {
    "fileses": [
        ("None"         , [f"projection/{pth}/fgmres/n{n}/noprec"      for n in ns]),
        ("Smoother only", [f"projection/{pth}/fgmres/n{n}/{sm}"        for n in ns]),
        ("Exact"        , [f"projection/{pth}/fgmres/n{n}/A-DEF1-{sm}" for n in ns]),
        ("DeepONet"     , [f"diffusion-dataset-3/deeponet/{pth}/fgmres/n{n}/A-DEF1-{sm}"   for n in ns]),
        ("RINO"         , [f"diffusion-dataset-3/rino/{pth}/fgmres/n{n}/A-DEF1-{sm}"       for n in ns]),
        ("VarMiON"      , [f"diffusion-dataset-3/varmion/{pth}/fgmres/n{n}/A-DEF1-{sm}"    for n in ns]),
        ("NGO"          , [f"diffusion-dataset-3/ngo/{pth}/fgmres/n{n}/A-DEF1-{sm}"       for n in ns]),
    ],
    "colours": [colors[kind] for kind in kinds],
    "alphas": [alphas[kind] for kind in kinds],
    "seps": [3],
    "outfile": "diffusion/barchart-FGMRES-dataset-3",
    "solver": "F-GMRES",
    "title": "",
}
makeplot(ns, **cfg)
cfg['outfile'] = "diffusion/barchart-FMGRES-dataset-3-withlegend"
cfg['loc'] = 'upper center'
makeplot(ns, **cfg)


pth1 = f"precondition/diffusion-dataset-3-9000-10000"
pth2 = f"precondition/with-dirichlet-basis/diffusion-dataset-3-9000-10000"
kinds = ["noprec", "smooth", "exact", "deeponet", "rino", "varmion", "ngo"]
cfg = {
    "fileses": [
        ("None"         , [f"projection/{pth1}/fgmres/n{n}/noprec"      for n in ns]),
        ("Smoother only", [f"projection/{pth1}/fgmres/n{n}/{sm}"        for n in ns]),
        ("Exact"        , [f"projection/{pth2}/fgmres/n{n}/A-DEF1-{sm}" for n in ns]),
        ("DeepONet"     , [f"diffusion-dataset-3/deeponet/{pth2}/fgmres/n{n}/A-DEF1-{sm}"   for n in ns]),
        ("RINO"         , [f"diffusion-dataset-3/rino/{pth2}/fgmres/n{n}/A-DEF1-{sm}"       for n in ns]),
        ("VarMiON"      , [f"diffusion-dataset-3/varmion/{pth2}/fgmres/n{n}/A-DEF1-{sm}"    for n in ns]),
        ("NGO"          , [f"diffusion-dataset-3/ngo/{pth2}/fgmres/n{n}/A-DEF1-{sm}"       for n in ns]),
    ],
    "colours": [colors[kind] for kind in kinds],
    "alphas": [alphas[kind] for kind in kinds],
    "seps": [3],
    "outfile": "diffusion/barchart-FGMRES-dataset-3-withlr",
    "solver": "F-GMRES",
    "title": "",
}
makeplot(ns, **cfg)
