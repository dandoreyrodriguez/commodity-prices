import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np


# ============================================================
# timing
# ============================================================

@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {end - start:.4f} sec")


# ============================================================
# filesystem
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# ============================================================
# grids + indexing
# ============================================================

def nearest_index(grid, value):
    return int(np.argmin(np.abs(grid - value)))


def quantile_indices(grid, probs):
    return [
        nearest_index(grid, np.quantile(grid, p))
        for p in probs
    ]


def curved_grid(x_max, n, curvature=3.0):
    x = np.linspace(0.0, 1.0, n)
    return x_max * x**curvature


# ============================================================
# plotting helpers
# ============================================================

def visible_ylim(
    model,
    X,
    q_ids=None,
    z_ids=None,
    s_zoom=None,
    pad=0.06,
):
    """
    Compute y-limits using only visible region.
    Useful for shared-y panel plots.
    """

    s_mask = (
        model.s_grid <= s_zoom
        if s_zoom is not None
        else np.ones_like(model.s_grid, dtype=bool)
    )

    vals = []

    if q_ids is not None and z_ids is not None:
        for i_q in q_ids:
            for i_z in z_ids:
                vals.append(X[s_mask, i_q, i_z])

    vals = np.concatenate(vals)

    ymin = np.nanmin(vals)
    ymax = np.nanmax(vals)

    gap = pad * (ymax - ymin if ymax > ymin else 1.0)

    return max(0.0, ymin - gap), ymax + gap


def symmetric_limits(X, q_clip=(0.01, 0.99)):
    vals = np.asarray(X).ravel()
    vals = vals[np.isfinite(vals)]

    lo, hi = np.quantile(vals, q_clip)

    m = max(abs(lo), abs(hi))

    return -m, m