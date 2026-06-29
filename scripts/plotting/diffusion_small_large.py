# Create bar charts

from .tables import makeplot
from .styles import colors, alphas, styles

ns = [20, 40, 80, 160, 320]
sm1 = "G_diffusion_0.1_1_1.5"
sm2 = "G_diffusion_0.05_1_1.5"
sm3 = "G_diffusion_0.025_1_1.5"

p1 = "projection-6x6"
p2 = "projection"
p3 = "projection-24x24"

m1 = "ngo-6x6"
m2 = "ngo"
m3 = "ngo-24x24"

# fmt: off
pth = f"precondition/diffusion-dataset-3-9000-10000"
kinds = ["smooth", "exact", "ngo"] * 3
cfg = {
    "fileses": [
        ("(6x6) Smoother only"  , [f"{p1}/{pth}/fgmres/n{n}/{sm1}"        for n in ns]),
        ("(6x6) Exact"          , [f"{p1}/{pth}/fgmres/n{n}/A-DEF1-{sm1}" for n in ns]),
        ("(6x6) NGO"            , [f"diffusion-dataset-3/{m1}/{pth}/fgmres/n{n}/A-DEF1-{sm1}" for n in ns]),
        ("(12x12) Smoother only", [f"{p2}/{pth}/fgmres/n{n}/{sm2}"        for n in ns]),
        ("(12x12) Exact"        , [f"{p2}/{pth}/fgmres/n{n}/A-DEF1-{sm2}" for n in ns]),
        ("(12x12) NGO"          , [f"diffusion-dataset-3/{m2}/{pth}/fgmres/n{n}/A-DEF1-{sm2}" for n in ns]),
        ("(24x24) Smoother only", [f"{p3}/{pth}/fgmres/n{n}/{sm3}"        for n in ns]),
        ("(24x24) Exact"        , [f"{p3}/{pth}/fgmres/n{n}/A-DEF1-{sm3}" for n in ns]),
        ("(24x24) NGO"          , [f"diffusion-dataset-3/{m3}/{pth}/fgmres/n{n}/A-DEF1-{sm3}" for n in ns]),
    ],
    "colours": [colors[kind] for kind in kinds],
    "alphas": [alphas[kind] for kind in kinds],
    "seps": [3, 6],
    "outfile": "diffusion/barchart-FGMRES-dataset-3-small-large",
    "solver": "F-GMRES",
    "title": "",
    "loc": "upper center",
    "lgnd": {"prop": {"size": 9}, "ncols": 3},
}
makeplot(ns, **cfg)
