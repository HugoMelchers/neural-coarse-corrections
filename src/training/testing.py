from pathlib import Path

from tqdm import tqdm
import torch
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt
import seaborn as sb

from ..dataset import DataSet
from ..utils.misc import save, roll_avg
from ..discretise import discretise_and_test
from ..utils.misc import relative_errors, nr_params
from ..utils.quadrature import get_quadrature


def test_models(data, table_fmt="markdown", loss_smoothing=10, boxplot=True, density_plot=True):
    load_results(data)
    add_errors(data)
    add_error_stats(data, keys="all")
    print_error_table(data, table_fmt)

    plot_history(data, loss_smoothing)
    plot_history(data, loss_smoothing, xlog=True)

    plot_separate_histories(data, smoothing=loss_smoothing)
    plot_separate_histories(data, smoothing=loss_smoothing, xlog=True)

    if boxplot:
        make_error_boxplots(data, logscale=True)

    if density_plot:
        make_error_densities(data, logscale=False)


def get_error(model, dataset, device="cpu", batch_size=20, desc=None):
    model.to(device)
    model.eval()
    N = len(dataset)
    if desc is not None:
        pb = tqdm(total=N, desc=desc, ascii=True, dynamic_ncols=True)

    indexloader = data.DataLoader(range(N), batch_size=batch_size, shuffle=False)
    errs = []
    with torch.no_grad():
        for indices in indexloader:
            batch = dataset.get_batch(indices, device)
            output = model(batch)
            rel_errs = relative_errors(output, batch["u"], batch["u/weights"]).sqrt()
            errs.append(rel_errs)
            if desc is not None:
                pb.update(batch_size)

    if desc is not None:
        pb.close()

    errs = torch.cat(errs)
    return errs.to("cpu")


def get_outputs(model, dataset, device="cpu"):
    model.to(device)
    model.eval()
    indexloader = DataLoader(range(len(dataset)), batch_size=100, shuffle=False)
    outputs = []
    with torch.no_grad():
        for indices in indexloader:
            batch = dataset.get_batch(indices, device)
            output = model(batch).to("cpu")
            outputs.append(output)

    outputs = torch.cat(outputs)
    return outputs


def load_results(data):
    for entry in data:
        path = entry["path"]
        if path is None:
            continue
        try:
            entry["model"] = torch.load(
                f"output/{path}/model.pt", map_location="cpu", weights_only=False
            )
        except FileNotFoundError as _:
            # Don't load model if the file doesn't exist
            print(f"File output/{path}/model.pt doesn't exist")
        try:
            history = torch.load(f"output/{path}/history.pt")
            entry["history/train"] = history["train"]
            entry["history/validate"] = history["validate"]
        except:
            # Don't load history if the history file doesn't exist
            pass


def add_errors(data):
    """
    Load or compute the errors of each model on the corresponding data sets. For example, if the
    first entry has a key-value pair `"data/test": "training-data/filename.pt:0-1000"`, then this
    function will open the corresponding errors file (for example for "output/model.pt", this would
    be "output/errors.pt"), and looks for the key "filename.pt:0-1000" in the resulting dict. If it
    exists, then those errors are returned. Else, the errors are computed, and stored in that entry
    of the dict.

    Note that if data["path"] is None, then the data is not saved anywhere.
    """

    for entry in data:
        error_data = {}
        path = entry["path"]
        dtype = entry.get("dtype", torch.float32)
        error_path = path and f"output/{path}/errors.pt"
        save_at_end = False
        # if the file we use to store errors already exists, load it, and get errors from there if
        # they exist. Else, start from scratch
        if path is not None:
            try:
                error_data = torch.load(error_path)
            except FileNotFoundError:
                error_data = {}
                save_at_end = True

        errors = {}

        for key, config in entry.items():
            if not key.startswith("data/"):
                continue

            # dataset, quad_name, quad = config
            dataset = config[0]
            quad_name = config[1]
            quad = config[2] if len(config) >= 3 else get_quadrature(quad_name)
            cache_key = dataset + ":" + quad_name

            testcase = key[5:]
            if cache_key in error_data:
                errs = error_data[cache_key]
            else:
                if "model" not in entry:
                    raise KeyError(f"entry with name \"{entry['name']}\" has no stored model")
                errs = discretise_and_test(
                    DataSet.load(dataset),
                    entry["model"],
                    quad_x=quad,
                    device="cpu",
                    batch_size=20,
                    desc="",
                    dtype=dtype,
                )
                error_data[cache_key] = errs
                save_at_end = True

            errors[f"errors/{testcase}"] = errs

        entry.update(errors)
        if path is not None and save_at_end:
            save(error_data, error_path, allow_exists=True)


def plot_history(data, smoothing=1, *, xlog=False, ylog=True):
    filename = "plots/convergence-history-train"
    for entry in data:
        if "history/train" not in entry:
            continue
        history = entry["history/train"]
        xs = torch.arange(smoothing, history.numel() + 1)
        plt.plot(xs, roll_avg(history, smoothing), label=entry["name"])

    if xlog:
        plt.xscale("log")
        filename += "-xlog"

    if ylog:
        plt.yscale("log")
        filename += "-ylog"

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Mean relative error (train)")
    plt.title("Convergence history")
    plt.savefig(
        filename + ".pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close()

    filename = "plots/convergence-history-validate"
    for entry in data:
        if "history/validate" not in entry:
            continue
        history = entry["history/validate"]
        xs = torch.arange(smoothing, history.numel() + 1)
        plt.semilogy(xs, roll_avg(history, smoothing), label=entry["name"])

    if xlog:
        plt.xscale("log")
        filename += "-xlog"

    if ylog:
        plt.yscale("log")
        filename += "-ylog"

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Mean relative error (validate)")
    plt.title("Convergence history")
    plt.savefig(
        filename + ".pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close()


def plot_separate_histories(data, smoothing=1, *, xlog=False, ylog=True):
    for entry in data:
        if "history/train" not in entry:
            continue

        path = entry["path"]
        suffix = "-xlog" if xlog else ""
        suffix += "-ylog" if ylog else ""
        filename = Path(f"output/{path}/convergence-history{suffix}.pdf")
        if filename.exists():
            continue

        hist_train = entry["history/train"]
        hist_val = entry.get("history/validate", None)
        xs = torch.arange(smoothing, hist_train.numel() + 1)
        plt.semilogy(xs, roll_avg(hist_train, smoothing), label="Train")
        if hist_val is not None:
            xs = torch.arange(smoothing, hist_val.numel() + 1)
            plt.semilogy(xs, roll_avg(hist_val, smoothing), label="Validate")

        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Mean relative error")
        if xlog:
            plt.xscale("log")
        if ylog:
            plt.yscale("log")
        plt.savefig(
            filename,
            transparent=True,
            bbox_inches="tight",
        )
        plt.close()


def add_error_stats(data, keys=None):
    maps = {
        "mean": lambda errs: errs.mean(),
        "rms": lambda errs: errs.square().mean().sqrt(),
        "std": lambda errs: errs.std(),
        "min": lambda errs: errs.min(),
        "max": lambda errs: errs.max(),
        "median": lambda errs: errs.median(),
        "5%": lambda errs: torch.sort(errs).values[round((errs.numel() - 1) * 5 / 100)],
        "95%": lambda errs: torch.sort(errs).values[round((errs.numel() - 1) * 95 / 100)],
    }

    if keys is None:
        keys = {"mean", "std"}
    elif keys == "all":
        keys = set(maps.keys())

    for entry in data:
        cases = {key[7:] for key in entry.keys() if key.startswith("errors/")}
        for case in cases:
            errs = entry[f"errors/{case}"]
            stats = {key: maps[key](errs).item() for key in keys}
            entry[f"error-stats/{case}"] = stats


def print_error_stats(data, cases=None, keys=None):
    if cases is None:
        cases = [key[5:] for key in data[0].keys() if key.startswith("data/")]

    names = {
        "mean": "Mean  ",
        "rms": "RMS   ",
        "std": "Std   ",
        "min": "Min   ",
        "max": "Max   ",
        "median": "Median",
        "5%": "5%    ",
        "95%": "95%   ",
    }

    if keys is None:
        keys = ["mean", "std"]
    elif keys == "all":
        keys = ["mean", "std", "rms", "min", "5%", "median", "95%", "max"]

    print("Errors (", " / ".join(cases), "):", sep="")
    for entry in data:
        name = entry["name"]
        params = nr_params(entry["model"]) if "model" in entry else 0
        stats = [entry[f"error-stats/{case}"] for case in cases]
        print(f"  {name} ({params}):")
        for key in keys:
            line = " / ".join(f"{s[key]:.6f}" for s in stats)
            print(f"    {names[key]}: {line}")

        print()


def _fmt_error_table_latex(rows, fmt):
    dot = lambda s: s.index(".") if "." in s else len(s)
    n_rows = len(rows)
    n_cols = len(rows[0])
    n_cases = n_cols - 2
    result = r"\begin{tabular}{" + " | ".join("l" for _ in range(n_cols)) + "}\n"
    result += "    \\toprule \n"
    result += rf"    \multirow{{2}}*{{Model}} & \multirow{{2}}*{{Parameters}} "
    result += rf"& \multicolumn{{{n_cases}}}{{c}}{{Test error}} \\"
    result += " \n"
    cases = rows[0][2:]

    # Figure out the amount of padding for the numbers
    for col in range(2, n_cols):
        m, s = 0, 0
        for row in rows[2:]:
            m_i, s_i = row[col]
            m = max(m, dot(m_i))
            s = max(s, dot(s_i))

        for row in rows[1:]:
            m_i, s_i = row[col]
            m_i = " " * (m - dot(m_i)) + m_i
            s_i = " " * (s - dot(s_i)) + s_i
            if fmt == "pm":
                row[col] = f"\\({m_i}\\% \\pm {s_i}\\%\\)"
            elif fmt == "-":
                row[col] = f"\\({m_i}\\% - {s_i}\\%\\)"

    # Align all the columns
    for col in range(n_cols):
        left = col == 0
        maxlen = max(len(row[col]) for row in rows[1:])
        for row in rows[1:]:
            row[col] = row[col].ljust(maxlen) if left else row[col].rjust(maxlen)

    result += rf"    & & " + " & ".join(cases) + " \\\\\n    \\midrule" + "\n"
    for i in range(1, n_rows):
        row = rows[i]
        result += "    " + " & ".join(row) + " \\\\\n"

    result += "    \\bottomrule\n\\end{tabular}"
    return result


def _fmt_error_table_markdown(rows, fmt):
    dot = lambda s: s.index(".") if "." in s else len(s)
    n_cols = len(rows[0])
    rows.insert(1, [":---"] + ["---:" for _ in rows[0][1:]])

    # Figure out the amount of padding for the numbers
    for col in range(2, n_cols):
        m, s = 0, 0
        for row in rows[2:]:
            m_i, s_i = row[col]
            m = max(m, dot(m_i))
            s = max(s, dot(s_i))

        for row in rows[2:]:
            m_i, s_i = row[col]
            m_i = " " * (m - dot(m_i)) + m_i
            s_i = " " * (s - dot(s_i)) + s_i
            if fmt == "pm":
                row[col] = f"{m_i}% ± {s_i}%"
            elif fmt == "-":
                row[col] = f"{m_i}% - {s_i}%"

    # Align all the columns
    for col in range(n_cols):
        left = col == 0
        maxlen = max(len(row[col]) for row in rows)
        for i, row in enumerate(rows):
            pad = "-" if i == 1 else " "
            if left:
                row[col] = row[col].ljust(maxlen, pad)
            else:
                row[col] = row[col].rjust(maxlen, pad)

    result = ""
    for row in rows:
        result += " | ".join(row)
        result += "\n"

    return result


def print_error_table(data, fmt, cases=None):
    if cases is None:
        cases = [key[12:] for key in data[0].keys() if key.startswith("error-stats/")]
    nr_cases = len(cases)
    # Columns: model name / #parameters / error...
    # Rows: header / models...
    n_cols = 2 + nr_cases
    n_rows = 1 + len(data)
    # Create table row by row
    rows = [[None] * n_cols for _ in range(n_rows)]
    rows[0] = ["Model", "Parameters"] + cases
    for _i, entry in enumerate(data):
        i = _i + 1
        rows[i][0] = entry["name"]
        rows[i][1] = str(nr_params(entry["model"]))
        for _j, case in enumerate(cases):
            j = _j + 2
            mean = 100 * entry[f"error-stats/{case}"]["mean"]
            std = 100 * entry[f"error-stats/{case}"]["std"]
            rows[i][j] = (f"{mean:.2f}", f"{std:.2f}")

    if fmt == "latex":
        result = _fmt_error_table_latex(rows, "pm")
    elif fmt == "markdown":
        result = _fmt_error_table_markdown(rows, "pm")
    print(result)


def print_error_table_5_95(data, fmt, cases=None):
    if cases is None:
        cases = [key[12:] for key in data[0].keys() if key.startswith("error-stats/")]
    nr_cases = len(cases)
    # Columns: model name / #parameters / error...
    # Rows: header / models...
    n_cols = 2 + nr_cases
    n_rows = 1 + len(data)
    # Create table row by row
    rows = [[None] * n_cols for _ in range(n_rows)]
    rows[0] = ["Model", "Parameters"] + cases
    for _i, entry in enumerate(data):
        i = _i + 1
        rows[i][0] = entry["name"]
        rows[i][1] = str(nr_params(entry["model"]))
        for _j, case in enumerate(cases):
            j = _j + 2
            p05 = 100 * entry[f"error-stats/{case}"]["5%"]
            p95 = 100 * entry[f"error-stats/{case}"]["95%"]
            rows[i][j] = (f"{p05:.2f}", f"{p95:.2f}")

    if fmt == "latex":
        result = _fmt_error_table_latex(rows, "-")
    elif fmt == "markdown":
        result = _fmt_error_table_markdown(rows, "-")
    print(result)


def make_error_boxplots(data, *, logscale=False, cases=None, titles=None):
    if cases is None:
        cases = [key[5:] for key in data[0].keys() if key.startswith("data/")]

    for i, case in enumerate(cases):
        errs = [entry[f"errors/{case}"] for entry in data]
        labels = [entry["name"] for entry in data]
        plt.boxplot(errs, labels=labels, vert=False)
        # plt.xticks(rotation=90)
        plt.yticks()
        if logscale:
            plt.xscale("log")

        if titles is not None:
            title = titles[i]
            plt.title(title)

        plt.savefig(
            f"plots/error-boxplot-{case}.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close()


def make_error_densities(data, *, logscale, cases=None, titles=None, xlims=None):
    if cases is None:
        cases = [key[5:] for key in data[0].keys() if key.startswith("data/")]

    for i, case in enumerate(cases):
        errs = [entry[f"errors/{case}"] for entry in data]
        labels = [entry["name"] for entry in data]
        for err, label in zip(errs, labels):
            sb.kdeplot(x=err, label=label)
        plt.xlabel("Relative error")
        plt.legend()
        if xlims is not None:
            plt.xlim(xlims)

        if logscale:
            plt.xscale("log")

        if titles is not None:
            title = titles[i]
            plt.title(title)

        plt.savefig(
            f"plots/error-densities-{case}.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close()
