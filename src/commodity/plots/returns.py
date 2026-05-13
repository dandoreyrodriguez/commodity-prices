import matplotlib.pyplot as plt
import numpy as np

from commodity.utils import quantile_indices


# ============================================================
# labels
# ============================================================

def _holding_return_label(n, return_type):
    if return_type == "level":
        return rf"$E_t[F_{{t+1}}^{{({n-1})}}] - F_t^{{({n})}}$"

    if return_type == "percent":
        return (
            rf"$\left(E_t[F_{{t+1}}^{{({n-1})}}] "
            rf"- F_t^{{({n})}}\right)/F_t^{{({n})}}$"
        )

    raise ValueError("return_type must be 'level' or 'percent'.")


def _holding_return_title(n, return_type):
    obj = "payoff" if return_type == "level" else "return"
    return f"Expected one-period futures holding {obj}, n={n}"


def _s_mask(model, s_zoom):
    if s_zoom is None:
        return np.ones_like(model.s_grid, dtype=bool)
    return model.s_grid <= s_zoom


def _symmetric_limits(arrays):
    visible = np.concatenate([np.asarray(X).ravel() for X in arrays])
    vmax = np.nanmax(np.abs(visible))

    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    return -vmax, vmax


# ============================================================
# heatmaps: holding return over (s, q), fixed z
# ============================================================

def heatmaps_expected_holding_return_s_q_across_z(
    model,
    z_probs=(0.05, 0.50, 0.80),
    n=3,
    return_type="level",
    s_zoom=None,
    cmap="coolwarm",
    shared_scale=True,
):
    z_ids = quantile_indices(model.z_grid, z_probs)
    s_mask = _s_mask(model, s_zoom)

    surfaces = []
    stored = {}

    for i_z in z_ids:
        out = model.expected_holding_return_fixed_z(
            i_z=i_z,
            n=n,
            return_type=return_type,
        )

        X = out["object"]
        surfaces.append(X)
        stored[i_z] = out

    fig, axes = plt.subplots(
        1,
        len(z_ids),
        figsize=(5 * len(z_ids), 4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    if len(z_ids) == 1:
        axes = [axes]

    S, Q = np.meshgrid(model.s_grid, model.q_grid, indexing="ij")

    if shared_scale:
        vmin, vmax = _symmetric_limits([X[s_mask, :] for X in surfaces])
    else:
        vmin = vmax = None

    last_im = None

    for ax, X, i_z, z_prob in zip(axes, surfaces, z_ids, z_probs):
        X_plot = X[s_mask, :]
        S_plot = S[s_mask, :]
        Q_plot = Q[s_mask, :]

        if shared_scale:
            im = ax.pcolormesh(
                S_plot,
                Q_plot,
                X_plot,
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            local_vmin, local_vmax = _symmetric_limits([X_plot])
            im = ax.pcolormesh(
                S_plot,
                Q_plot,
                X_plot,
                shading="auto",
                cmap=cmap,
                vmin=local_vmin,
                vmax=local_vmax,
            )

        last_im = im

        ax.axvline(0.0, linewidth=0.8)
        ax.set_title(rf"$z_{{{z_prob:.2f}}}={model.z_grid[i_z]:.4f}$")
        ax.set_xlabel(r"storage today, $s$")

    axes[0].set_ylabel(r"inflow, $q$")

    fig.colorbar(
        last_im,
        ax=axes,
        shrink=0.85,
        label=_holding_return_label(n, return_type),
    )

    fig.suptitle(_holding_return_title(n, return_type), y=1.05)

    return fig, stored


# ============================================================
# lines: holding return against s, across q, fixed z
# ============================================================

def lines_expected_holding_return_vary_s_across_z(
    model,
    z_probs=(0.05, 0.50, 0.80),
    q_probs=(0.05, 0.50, 0.80),
    n=3,
    return_type="level",
    s_zoom=None,
):
    z_ids = quantile_indices(model.z_grid, z_probs)
    q_ids = quantile_indices(model.q_grid, q_probs)

    fig, axes = plt.subplots(
        1,
        len(z_ids),
        figsize=(5 * len(z_ids), 4),
        sharey=True,
        constrained_layout=True,
    )

    if len(z_ids) == 1:
        axes = [axes]

    stored = {}

    for ax, i_z, z_prob in zip(axes, z_ids, z_probs):
        out = model.expected_holding_return_fixed_z(
            i_z=i_z,
            n=n,
            return_type=return_type,
        )

        X = out["object"]
        stored[i_z] = out

        for i_q, q_prob in zip(q_ids, q_probs):
            ax.plot(
                model.s_grid,
                X[:, i_q],
                label=rf"$q_{{{q_prob:.2f}}}={model.q_grid[i_q]:.4f}$",
            )

        ax.axhline(0.0, linewidth=1.0, linestyle="--")
        ax.set_title(rf"$z_{{{z_prob:.2f}}}={model.z_grid[i_z]:.4f}$")
        ax.set_xlabel(r"storage today, $s$")

        if s_zoom is not None:
            ax.set_xlim(0.0, s_zoom)

    axes[0].set_ylabel(_holding_return_label(n, return_type))

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(q_ids),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.suptitle(_holding_return_title(n, return_type), y=1.05)

    return fig, stored


# ============================================================
# heatmaps: holding return over (s, z), fixed q
# ============================================================

def heatmaps_expected_holding_return_s_z_across_q(
    model,
    q_probs=(0.05, 0.50, 0.80),
    n=3,
    return_type="level",
    s_zoom=None,
    cmap="coolwarm",
    shared_scale=True,
):
    q_ids = quantile_indices(model.q_grid, q_probs)
    s_mask = _s_mask(model, s_zoom)

    surfaces = []
    stored = {}

    for i_q in q_ids:
        out = model.expected_holding_return_fixed_q(
            i_q=i_q,
            n=n,
            return_type=return_type,
        )

        X = out["object"]
        surfaces.append(X)
        stored[i_q] = out

    fig, axes = plt.subplots(
        1,
        len(q_ids),
        figsize=(5 * len(q_ids), 4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    if len(q_ids) == 1:
        axes = [axes]

    S, Z = np.meshgrid(model.s_grid, model.z_grid, indexing="ij")

    if shared_scale:
        vmin, vmax = _symmetric_limits([X[s_mask, :] for X in surfaces])
    else:
        vmin = vmax = None

    last_im = None

    for ax, X, i_q, q_prob in zip(axes, surfaces, q_ids, q_probs):
        X_plot = X[s_mask, :]
        S_plot = S[s_mask, :]
        Z_plot = Z[s_mask, :]

        if shared_scale:
            im = ax.pcolormesh(
                S_plot,
                Z_plot,
                X_plot,
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            local_vmin, local_vmax = _symmetric_limits([X_plot])
            im = ax.pcolormesh(
                S_plot,
                Z_plot,
                X_plot,
                shading="auto",
                cmap=cmap,
                vmin=local_vmin,
                vmax=local_vmax,
            )

        last_im = im

        ax.set_title(rf"$q_{{{q_prob:.2f}}}={model.q_grid[i_q]:.4f}$")
        ax.set_xlabel(r"storage today, $s$")

    axes[0].set_ylabel(r"productivity, $z$")

    fig.colorbar(
        last_im,
        ax=axes,
        shrink=0.85,
        label=_holding_return_label(n, return_type),
    )

    fig.suptitle(_holding_return_title(n, return_type), y=1.05)

    return fig, stored


# ============================================================
# lines: holding return against s, across z, fixed q
# ============================================================

def lines_expected_holding_return_vary_s_across_q(
    model,
    q_probs=(0.05, 0.50, 0.80),
    z_probs=(0.05, 0.50, 0.80),
    n=3,
    return_type="level",
    s_zoom=None,
):
    q_ids = quantile_indices(model.q_grid, q_probs)
    z_ids = quantile_indices(model.z_grid, z_probs)

    fig, axes = plt.subplots(
        1,
        len(q_ids),
        figsize=(5 * len(q_ids), 4),
        sharey=True,
        constrained_layout=True,
    )

    if len(q_ids) == 1:
        axes = [axes]

    stored = {}

    for ax, i_q, q_prob in zip(axes, q_ids, q_probs):
        out = model.expected_holding_return_fixed_q(
            i_q=i_q,
            n=n,
            return_type=return_type,
        )

        X = out["object"]
        stored[i_q] = out

        for i_z, z_prob in zip(z_ids, z_probs):
            ax.plot(
                model.s_grid,
                X[:, i_z],
                label=rf"$z_{{{z_prob:.2f}}}={model.z_grid[i_z]:.4f}$",
            )

        ax.axhline(0.0, linewidth=1.0, linestyle="--")
        ax.set_title(rf"$q_{{{q_prob:.2f}}}={model.q_grid[i_q]:.4f}$")
        ax.set_xlabel(r"storage today, $s$")

        if s_zoom is not None:
            ax.set_xlim(0.0, s_zoom)

    axes[0].set_ylabel(_holding_return_label(n, return_type))

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(z_ids),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.suptitle(_holding_return_title(n, return_type), y=1.05)

    return fig, stored