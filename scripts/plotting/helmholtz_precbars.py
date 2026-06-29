# Create bar charts

from .tables import makeplot
from .styles import colors, alphas

ns = [20, 40, 80, 160, 320]


ref = "projection"
mdl = "ngo"
data = "helmholtz-9000-10000"
pth = f"precondition/{data}"
sm = "G_helmholtz_0.05_1_1.5"

# fmt: off
kinds = ["noprec", "smooth", "exact", "rino", "varmion", "ngo"]
cfg1 = {
    "fileses": [
        ("None"         , [f"projection/{pth}/fgmres/n{n}/noprec"            for n in ns]),
        ("Smoother only", [f"projection/{pth}/fgmres/n{n}/{sm}"              for n in ns]),
        ("Exact"        , [f"projection/{pth}/fgmres/n{n}/A-DEF1-{sm}" for n in ns]),
        # ("DeepONet"   , [f"helmholtz/deeponet/{pth}/fgmres/n{n}/A-DEF1-{sm}"    for n in ns]),
        ("RINO"         , [f"helmholtz/rino/{pth}/fgmres/n{n}/A-DEF1-{sm}"        for n in ns]),
        ("VarMiON"      , [f"helmholtz/varmion/{pth}/fgmres/n{n}/A-DEF1-{sm}"      for n in ns]),
        ("NGO"          , [f"helmholtz/ngo/{pth}/fgmres/n{n}/A-DEF1-{sm}"         for n in ns]),
    ],
    "colours": [colors[kind] for kind in kinds],
    "alphas": [alphas[kind] for kind in kinds],
    "seps": [3],
    "outfile": "helmholtz/barchart-FGMRES",
    "solver": "F-GMRES",
    "loc": "upper center",
    "title": "",
}
# fmt: on

makeplot(ns, **cfg1)
