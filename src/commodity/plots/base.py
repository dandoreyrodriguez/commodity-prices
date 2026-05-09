from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from commodity.utils import ensure_dir


def save_fig(fig, filename, folder="outputs/figures", tight=True):
    folder = Path(folder)
    ensure_dir(folder)

    if tight:
        fig.tight_layout()

    fig.savefig(folder / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_and_save(
    fig_func,
    filename,
    *args,
    folder="outputs/figures",
    tight=True,
    **kwargs,
):
    out = fig_func(*args, **kwargs)
    fig = out[0] if isinstance(out, tuple) else out

    save_fig(
        fig,
        filename,
        folder=folder,
        tight=tight,
    )


def heatmap_limits(
    X,
    q_clip=(0.01, 0.99),
    symmetric=True,
):
    vals = np.asarray(X).ravel()
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return None, None

    lo, hi = np.quantile(vals, q_clip)

    if symmetric:
        m = max(abs(lo), abs(hi))
        return -m, m

    return lo, hi