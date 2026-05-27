# Create bar charts

from src import *
from src.plotting import groupedbar
import matplotlib.pyplot as plt
from .tables import makeplot
from .styles import colors, alphas

ns = [20, 40, 80, 160, 320]


# fmt: off
# Two plots:
# 1. All models on low Péclet data
# 2. All models on high Péclet data

data1 = "advdiff-low-peclet-9000-10000"
data2 = "advdiff-high-peclet-9000-10000"
sm = "gs4"

kinds = ["noprec", "smooth", "exact", "deeponet", "rino", "varmion", "ngo"]
cfg1 = {
    "fileses": [
        ("None"         , [f"projection/precondition/{data1}/fgmres/n{n}/noprec"      for n in ns]),
        ("Smoother only", [f"projection/precondition/{data1}/fgmres/n{n}/{sm}"        for n in ns]),
        ("Exact"        , [f"projection/precondition/{data1}/fgmres/n{n}/A-DEF1-{sm}" for n in ns]),
        ("DeepONet"     , [f"advdiff/deeponet/precondition/{data1}/fgmres/n{n}/A-DEF1-{sm}"  for n in ns]),
        ("RINO"         , [f"advdiff/rino/precondition/{data1}/fgmres/n{n}/A-DEF1-{sm}"      for n in ns]),
        ("VarMiON"      , [f"advdiff/varmion/precondition/{data1}/fgmres/n{n}/A-DEF1-{sm}"    for n in ns]),
        ("NGO"          , [f"advdiff/ngo/precondition/{data1}/fgmres/n{n}/A-DEF1-{sm}"       for n in ns]),
    ],
    "colours": [colors[kind] for kind in kinds],
    "alphas": [alphas[kind] for kind in kinds],
    "seps": [3],
    "outfile": "advdiff/barchart-low-peclet",
    "solver": "F-GMRES",
    "loc": "upper center",
    "title": "",
}
makeplot(ns, **cfg1)

cfg2 = {
    "fileses": [
        ("None"         , [f"projection/precondition/{data2}/fgmres/n{n}/noprec"      for n in ns]),
        ("Smoother only", [f"projection/precondition/{data2}/fgmres/n{n}/{sm}"        for n in ns]),
        ("Exact"        , [f"projection/precondition/{data2}/fgmres/n{n}/A-DEF1-{sm}" for n in ns]),
        ("DeepONet"     , [f"advdiff/deeponet/precondition/{data2}/fgmres/n{n}/A-DEF1-{sm}"  for n in ns]),
        ("RINO"         , [f"advdiff/rino/precondition/{data2}/fgmres/n{n}/A-DEF1-{sm}"      for n in ns]),
        ("VarMiON"      , [f"advdiff/varmion/precondition/{data2}/fgmres/n{n}/A-DEF1-{sm}"    for n in ns]),
        ("NGO"          , [f"advdiff/ngo/precondition/{data2}/fgmres/n{n}/A-DEF1-{sm}"       for n in ns]),
    ],
    "colours": [colors[kind] for kind in kinds],
    "alphas": [alphas[kind] for kind in kinds],
    "seps": [3],
    "outfile": "advdiff/barchart-high-peclet",
    "solver": "F-GMRES",
    "title": "",
}
makeplot(ns, **cfg2)
