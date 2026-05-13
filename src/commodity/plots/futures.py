import matplotlib.pyplot as plt
import numpy as np

from commodity.diagnostics import (
    expected_spot_change_fixed_q,
    expected_spot_change_fixed_z,
    wedge_surface_fixed_q,
    wedge_surface_fixed_z,
)
from commodity.plots.base import heatmap_limits
from commodity.utils import nearest_index, quantile_indices


def _as_list(axes):
    return axes if isinstance(axes, (list, np.ndarray)) else [axes]


def _outside_legend(fig, ax, ncol=3):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=ncol,
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(0.5, 0.98),
        )


def futures_curves_vary_q_across_z(
    model,
    q_probs=(0.05, 0.50, 0.80),
    z_probs=(0.05, 0.50, 0.80),
    s_prob=0.50,
    T=12,
):
    i_s = nearest_index(model.s_grid, np.quantile(model.s_grid, s_prob))
    q_ids = quantile_indices(model.q_grid, q_probs)
    z_ids = quantile_indices(model.z_grid, z_probs)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(q_ids)))

    fig, axes = plt.subplots(
        1,
        len(z_ids),
        figsize=(5 * len(z_ids), 4),
        sharey=True,
    )
    axes = _as_list(axes)

    stored = {}

    for ax, i_z, z_prob in zip(axes, z_ids, z_probs):
        stored[i_z] = {}

        for color, i_q, q_prob in zip(colors, q_ids, q_probs):
            out = model.futures_curve_at_index(i_s, i_q, i_z, T=T)
            stored[i_z][i_q] = out

            ax.plot(
                out["maturity"],
                out["F"],
                color=color,
                marker="o",
                label=rf"$F$, $q_{{{q_prob:.2f}}}$",
            )
            ax.plot(
                out["maturity"],
                out["Ep"],
                color=color,
                linestyle="--",
                marker="s",
                label=rf"$E[p]$, $q_{{{q_prob:.2f}}}$",
            )

        ax.set_title(rf"$z_{{{z_prob:.2f}}}$")
        ax.set_xlabel("maturity")

    axes[0].set_ylabel("price")
    _outside_legend(fig, axes[0], ncol=max(1, 2 * len(q_ids)))

    fig.suptitle(
        rf"Futures vs expected spot, fixed "
        rf"$s_{{{s_prob:.2f}}}={model.s_grid[i_s]:.4f}$"
    )
    fig.subplots_adjust(top=0.78)

    return fig, axes, stored


def futures_curves_vary_z_across_q(
    model,
    q_probs=(0.05, 0.50, 0.80),
    z_probs=(0.05, 0.50, 0.80),
    s_prob=0.50,
    T=12,
):
    i_s = nearest_index(model.s_grid, np.quantile(model.s_grid, s_prob))
    q_ids = quantile_indices(model.q_grid, q_probs)
    z_ids = quantile_indices(model.z_grid, z_probs)

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(z_ids)))

    fig, axes = plt.subplots(
        1,
        len(q_ids),
        figsize=(5 * len(q_ids), 4),
        sharey=True,
    )
    axes = _as_list(axes)

    stored = {}

    for ax, i_q, q_prob in zip(axes, q_ids, q_probs):
        stored[i_q] = {}

        for color, i_z, z_prob in zip(colors, z_ids, z_probs):
            out = model.futures_curve_at_index(i_s, i_q, i_z, T=T)
            stored[i_q][i_z] = out

            ax.plot(
                out["maturity"],
                out["F"],
                color=color,
                marker="o",
                label=rf"$F$, $z_{{{z_prob:.2f}}}$",
            )
            ax.plot(
                out["maturity"],
                out["Ep"],
                color=color,
                linestyle="--",
                marker="s",
                label=rf"$E[p]$, $z_{{{z_prob:.2f}}}$",
            )

        ax.set_title(rf"$q_{{{q_prob:.2f}}}$")
        ax.set_xlabel("maturity")

    axes[0].set_ylabel("price")
    _outside_legend(fig, axes[0], ncol=max(1, 2 * len(z_ids)))

    fig.suptitle(
        rf"Futures vs expected spot, fixed "
        rf"$s_{{{s_prob:.2f}}}={model.s_grid[i_s]:.4f}$"
    )
    fig.subplots_adjust(top=0.78)

    return fig, axes, stored


def wedge_heatmaps_s_q_across_z(
    model,
    T=12,
    z_probs=(0.05, 0.50, 0.80),
    s_zoom=None,
    q_clip=(0.01, 0.99),
):
    z_ids = quantile_indices(model.z_grid, z_probs)
    W_list = [wedge_surface_fixed_z(model, i_z, T=T) for i_z in z_ids]

    s_mask = (
        model.s_grid <= s_zoom
        if s_zoom is not None
        else np.ones_like(model.s_grid, dtype=bool)
    )
    W_visible = np.stack([W[s_mask, :] for W in W_list], axis=-1)

    vmin, vmax = heatmap_limits(W_visible, q_clip=q_clip, symmetric=True)

    S, Q = np.meshgrid(model.s_grid, model.q_grid, indexing="ij")

    fig, axes = plt.subplots(
        1,
        len(z_ids),
        figsize=(5 * len(z_ids), 4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = _as_list(axes)

    ims = []

    for ax, W, i_z, z_prob in zip(axes, W_list, z_ids, z_probs):
        im = ax.pcolormesh(
            S,
            Q,
            W,
            shading="auto",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        ims.append(im)

        ax.axvline(0.0, linewidth=0.8)
        ax.set_title(rf"$z_{{{z_prob:.2f}}}={model.z_grid[i_z]:.4f}$")
        ax.set_xlabel(r"storage today, $s$")

        if s_zoom is not None:
            ax.set_xlim(0.0, s_zoom)

    axes[0].set_ylabel(r"inflow, $q$")

    cbar = fig.colorbar(ims[0], ax=axes, fraction=0.05, pad=0.04)
    cbar.set_label(rf"$F_{{t,t+{T}}}-E_t[p_{{t+{T}}}^s]$")

    fig.suptitle(rf"Futures wedge over $(s,q)$, maturity $T={T}$")

    return fig, axes


def wedge_heatmaps_s_z_across_q(
    model,
    T=12,
    q_probs=(0.05, 0.50, 0.80),
    s_zoom=None,
    q_clip=(0.01, 0.99),
):
    q_ids = quantile_indices(model.q_grid, q_probs)
    W_list = [wedge_surface_fixed_q(model, i_q, T=T) for i_q in q_ids]

    s_mask = (
        model.s_grid <= s_zoom
        if s_zoom is not None
        else np.ones_like(model.s_grid, dtype=bool)
    )
    W_visible = np.stack([W[s_mask, :] for W in W_list], axis=-1)

    vmin, vmax = heatmap_limits(W_visible, q_clip=q_clip, symmetric=True)

    S, Z = np.meshgrid(model.s_grid, model.z_grid, indexing="ij")

    fig, axes = plt.subplots(
        1,
        len(q_ids),
        figsize=(5 * len(q_ids), 4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = _as_list(axes)

    ims = []

    for ax, W, i_q, q_prob in zip(axes, W_list, q_ids, q_probs):
        im = ax.pcolormesh(
            S,
            Z,
            W,
            shading="auto",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        ims.append(im)

        ax.set_title(rf"$q_{{{q_prob:.2f}}}={model.q_grid[i_q]:.4f}$")
        ax.set_xlabel(r"storage today, $s$")

        if s_zoom is not None:
            ax.set_xlim(0.0, s_zoom)

    axes[0].set_ylabel(r"productivity, $z$")

    cbar = fig.colorbar(ims[0], ax=axes, fraction=0.05, pad=0.04)
    cbar.set_label(rf"$F_{{t,t+{T}}}-E_t[p_{{t+{T}}}^s]$")

    fig.suptitle(rf"Futures wedge over $(s,z)$, maturity $T={T}$")

    return fig, axes


def wedge_lines_vary_q_across_z(
    model,
    T=12,
    q_probs=(0.05, 0.50, 0.80),
    z_probs=(0.05, 0.50, 0.80),
    s_zoom=None,
):
    q_ids = quantile_indices(model.q_grid, q_probs)
    z_ids = quantile_indices(model.z_grid, z_probs)

    W_by_z = {i_z: wedge_surface_fixed_z(model, i_z, T=T) for i_z in z_ids}

    s_mask = (
        model.s_grid <= s_zoom
        if s_zoom is not None
        else np.ones_like(model.s_grid, dtype=bool)
    )

    vals = []
    for i_z in z_ids:
        W = W_by_z[i_z]
        for i_q in q_ids:
            vals.append(W[s_mask, i_q])

    vals = np.concatenate(vals)
    ymin = np.nanmin(vals)
    ymax = np.nanmax(vals)
    gap = 0.08 * (ymax - ymin if ymax > ymin else 1.0)

    fig, axes = plt.subplots(
        1,
        len(z_ids),
        figsize=(5 * len(z_ids), 4),
        sharex=True,
        sharey=True,
    )
    axes = _as_list(axes)

    for ax, i_z, z_prob in zip(axes, z_ids, z_probs):
        W = W_by_z[i_z]

        for i_q, q_prob in zip(q_ids, q_probs):
            ax.plot(
                model.s_grid,
                W[:, i_q],
                label=rf"$q_{{{q_prob:.2f}}}$",
            )

        ax.axhline(0.0, linewidth=0.8)
        ax.set_title(rf"$z_{{{z_prob:.2f}}}$")
        ax.set_xlabel(r"storage today, $s$")
        ax.set_ylim(ymin - gap, ymax + gap)

        if s_zoom is not None:
            ax.set_xlim(0.0, s_zoom)

    axes[0].set_ylabel(rf"$F_{{t,t+{T}}}-E_t[p_{{t+{T}}}^s]$")

    _outside_legend(fig, axes[0], ncol=len(q_ids))

    fig.suptitle(rf"Futures wedge lines, maturity $T={T}$")
    fig.subplots_adjust(top=0.78)

    return fig, axes


def wedge_term_structure_grid(
    model,
    s_probs=(0.01, 0.10, 0.50),
    q_probs=(0.05, 0.50, 0.80),
    z_probs=(0.05, 0.50, 0.80),
    T=12,
):
    s_ids = quantile_indices(model.s_grid, s_probs)
    q_ids = quantile_indices(model.q_grid, q_probs)
    z_ids = quantile_indices(model.z_grid, z_probs)

    fig, axes = plt.subplots(
        len(s_ids),
        len(z_ids),
        figsize=(5 * len(z_ids), 3.5 * len(s_ids)),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(len(s_ids), len(z_ids))

    stored = {}

    for r, (i_s, s_prob) in enumerate(zip(s_ids, s_probs)):
        stored[i_s] = {}

        for c, (i_z, z_prob) in enumerate(zip(z_ids, z_probs)):
            ax = axes[r, c]
            stored[i_s][i_z] = {}

            for i_q, q_prob in zip(q_ids, q_probs):
                out = model.futures_curve_at_index(i_s, i_q, i_z, T=T)
                stored[i_s][i_z][i_q] = out

                ax.plot(
                    out["maturity"],
                    out["wedge"],
                    marker="o",
                    label=rf"$q_{{{q_prob:.2f}}}$",
                )

            ax.axhline(0.0, linewidth=0.8)
            ax.set_title(
                rf"$s_{{{s_prob:.2f}}}={model.s_grid[i_s]:.4f}$, "
                rf"$z_{{{z_prob:.2f}}}={model.z_grid[i_z]:.4f}$"
            )
            ax.set_xlabel("maturity")

            if c == 0:
                ax.set_ylabel(r"$F-E[p]$")

    _outside_legend(fig, axes[0, 0], ncol=len(q_ids))

    fig.suptitle("Wedge term structures across scarcity states")
    fig.subplots_adjust(top=0.92)

    return fig, axes, stored


def expected_spot_heatmaps_s_q_across_z(
    model,
    T=12,
    z_probs=(0.05, 0.50, 0.80),
    s_zoom=None,
    change=True,
    pct=False,
    cmap="RdBu_r",
):
    z_ids = quantile_indices(model.z_grid, z_probs)

    X_list = []
    for i_z in z_ids:
        if change:
            X = expected_spot_change_fixed_z(model, i_z, T=T, pct=pct)
        else:
            X = model.futures_surface_fixed_z(i_z, T=T, object_name="Ep")
        X_list.append(X)

    s_mask = (
        model.s_grid <= s_zoom
        if s_zoom is not None
        else np.ones_like(model.s_grid, dtype=bool)
    )
    X_visible = np.stack([X[s_mask, :] for X in X_list], axis=-1)

    vmin, vmax = heatmap_limits(X_visible, symmetric=change)

    S, Q = np.meshgrid(model.s_grid, model.q_grid, indexing="ij")

    fig, axes = plt.subplots(
        1,
        len(z_ids),
        figsize=(5 * len(z_ids), 4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = _as_list(axes)

    ims = []

    for ax, X, i_z, z_prob in zip(axes, X_list, z_ids, z_probs):
        im = ax.pcolormesh(
            S,
            Q,
            X,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ims.append(im)

        ax.set_title(rf"$z_{{{z_prob:.2f}}}={model.z_grid[i_z]:.4f}$")
        ax.set_xlabel(r"storage today, $s$")

        if s_zoom is not None:
            ax.set_xlim(0.0, s_zoom)

    axes[0].set_ylabel(r"inflow, $q$")

    if change:
        label = rf"$E_t[p^s_{{t+{T}}}]-p^s_t$"
        title = rf"Expected spot price change over $(s,q)$, horizon $T={T}$"
        if pct:
            label = rf"$(E_t[p^s_{{t+{T}}}]-p^s_t)/p^s_t$"
    else:
        label = rf"$E_t[p^s_{{t+{T}}}]$"
        title = rf"Expected future spot price over $(s,q)$, horizon $T={T}$"

    cbar = fig.colorbar(ims[0], ax=axes, fraction=0.05, pad=0.04)
    cbar.set_label(label)

    fig.suptitle(title)

    return fig, axes


def expected_spot_heatmaps_s_z_across_q(
    model,
    T=12,
    q_probs=(0.05, 0.50, 0.80),
    s_zoom=None,
    change=True,
    pct=False,
    cmap="RdBu_r",
):
    q_ids = quantile_indices(model.q_grid, q_probs)

    X_list = []
    for i_q in q_ids:
        if change:
            X = expected_spot_change_fixed_q(model, i_q, T=T, pct=pct)
        else:
            X = model.futures_surface_fixed_q(i_q, T=T, object_name="Ep")
        X_list.append(X)

    s_mask = (
        model.s_grid <= s_zoom
        if s_zoom is not None
        else np.ones_like(model.s_grid, dtype=bool)
    )
    X_visible = np.stack([X[s_mask, :] for X in X_list], axis=-1)

    vmin, vmax = heatmap_limits(X_visible, symmetric=change)

    S, Z = np.meshgrid(model.s_grid, model.z_grid, indexing="ij")

    fig, axes = plt.subplots(
        1,
        len(q_ids),
        figsize=(5 * len(q_ids), 4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = _as_list(axes)

    ims = []

    for ax, X, i_q, q_prob in zip(axes, X_list, q_ids, q_probs):
        im = ax.pcolormesh(
            S,
            Z,
            X,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ims.append(im)

        ax.set_title(rf"$q_{{{q_prob:.2f}}}={model.q_grid[i_q]:.4f}$")
        ax.set_xlabel(r"storage today, $s$")

        if s_zoom is not None:
            ax.set_xlim(0.0, s_zoom)

    axes[0].set_ylabel(r"productivity, $z$")

    if change:
        label = rf"$E_t[p^s_{{t+{T}}}]-p^s_t$"
        title = rf"Expected spot price change over $(s,z)$, horizon $T={T}$"
        if pct:
            label = rf"$(E_t[p^s_{{t+{T}}}]-p^s_t)/p^s_t$"
    else:
        label = rf"$E_t[p^s_{{t+{T}}}]$"
        title = rf"Expected future spot price over $(s,z)$, horizon $T={T}$"

    cbar = fig.colorbar(ims[0], ax=axes, fraction=0.05, pad=0.04)
    cbar.set_label(label)

    fig.suptitle(title)

    return fig, axes


def pure_futures_term_structure_grid(
    model,
    s_probs=(0.01, 0.10, 0.50),
    q_probs=(0.01, 0.05, 0.50),
    z_probs=(0.05, 0.50, 0.80),
    T=12,
):
    s_ids = quantile_indices(model.s_grid, s_probs)
    q_ids = quantile_indices(model.q_grid, q_probs)
    z_ids = quantile_indices(model.z_grid, z_probs)

    fig, axes = plt.subplots(
        len(s_ids),
        len(z_ids),
        figsize=(5 * len(z_ids), 3.5 * len(s_ids)),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(len(s_ids), len(z_ids))

    stored = {}

    for r, (i_s, s_prob) in enumerate(zip(s_ids, s_probs)):
        stored[i_s] = {}

        for c, (i_z, z_prob) in enumerate(zip(z_ids, z_probs)):
            ax = axes[r, c]
            stored[i_s][i_z] = {}

            for i_q, q_prob in zip(q_ids, q_probs):
                out = model.futures_curve_at_index(i_s, i_q, i_z, T=T)
                stored[i_s][i_z][i_q] = out

                ax.plot(
                    out["maturity"],
                    out["F"],
                    marker="o",
                    label=rf"$q_{{{q_prob:.2f}}}$",
                )

            ax.axhline(
                model.price_s[i_s, q_ids[0], i_z],
                linewidth=0.8,
                linestyle="--",
                alpha=0.6,
            )
            ax.set_title(
                rf"$s_{{{s_prob:.2f}}}={model.s_grid[i_s]:.4f}$, "
                rf"$z_{{{z_prob:.2f}}}={model.z_grid[i_z]:.4f}$"
            )
            ax.set_xlabel("maturity")

            if c == 0:
                ax.set_ylabel(r"$F_{t,t+h}$")

    _outside_legend(fig, axes[0, 0], ncol=len(q_ids))

    fig.suptitle("Pure futures term structures: contango/backwardation")
    fig.subplots_adjust(top=0.92)

    return fig, axes, stored


def pure_expected_spot_term_structure_grid(
    model,
    s_probs=(0.01, 0.10, 0.50),
    q_probs=(0.01, 0.05, 0.50),
    z_probs=(0.05, 0.50, 0.80),
    T=12,
):
    s_ids = quantile_indices(model.s_grid, s_probs)
    q_ids = quantile_indices(model.q_grid, q_probs)
    z_ids = quantile_indices(model.z_grid, z_probs)

    fig, axes = plt.subplots(
        len(s_ids),
        len(z_ids),
        figsize=(5 * len(z_ids), 3.5 * len(s_ids)),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(len(s_ids), len(z_ids))

    stored = {}

    for r, (i_s, s_prob) in enumerate(zip(s_ids, s_probs)):
        stored[i_s] = {}

        for c, (i_z, z_prob) in enumerate(zip(z_ids, z_probs)):
            ax = axes[r, c]
            stored[i_s][i_z] = {}

            for i_q, q_prob in zip(q_ids, q_probs):
                out = model.futures_curve_at_index(i_s, i_q, i_z, T=T)
                stored[i_s][i_z][i_q] = out

                ax.plot(
                    out["maturity"],
                    out["Ep"],
                    marker="s",
                    label=rf"$q_{{{q_prob:.2f}}}$",
                )

            ax.axhline(
                model.price_s[i_s, q_ids[0], i_z],
                linewidth=0.8,
                linestyle="--",
                alpha=0.6,
            )
            ax.set_title(
                rf"$s_{{{s_prob:.2f}}}={model.s_grid[i_s]:.4f}$, "
                rf"$z_{{{z_prob:.2f}}}={model.z_grid[i_z]:.4f}$"
            )
            ax.set_xlabel("maturity")

            if c == 0:
                ax.set_ylabel(r"$E_t[p^s_{t+h}]$")

    _outside_legend(fig, axes[0, 0], ncol=len(q_ids))

    fig.suptitle("Expected spot term structures")
    fig.subplots_adjust(top=0.92)

    return fig, axes, stored