import matplotlib.pyplot as plt

plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.size": 11,
        "font.family": "lmodern",
    }
)

mycolors = {
    "Blue-fill": (0.855, 0.91, 0.988),
    "Blue-stroke": (0.424, 0.557, 0.749),
    "Green-fill": (0.9, 1, 0.9),
    "Green-stroke": (0.4, 0.9, 0.4),
    "Orange-fill": (1, 0.902, 0.8),
    "Orange-stroke": (0.843, 0.608, 0),
    "Purple-fill": (0.9, 0.85, 0.9),
    "Purple-stroke": (0.588, 0.451, 0.651),
    "Yellow-fill": (1.0, 0.949, 0.8),
    "Yellow-stroke": (0.839, 0.714, 0.337),
    "Red-fill": (0.973, 0.808, 0.8),
    "Red-stroke": (0.722, 0.329, 0.314),
}

styles = {
    "noprec": {"color": "black", "alpha": 0.7},
    "smooth": {"color": "black", "alpha": 0.5},
    "exact": {"color": "C0", "alpha": 0.7},
    "deeponet": {"color": "C1", "alpha": 0.7},
    "rino": {"color": "C2", "alpha": 0.7},
    "varmion": {"color": "C3", "alpha": 0.7},
    "ngo": {"color": "C4", "alpha": 0.7},
}

colors = {key: styles[key]["color"] for key in styles.keys()}
alphas = {key: styles[key]["alpha"] for key in styles.keys()}
