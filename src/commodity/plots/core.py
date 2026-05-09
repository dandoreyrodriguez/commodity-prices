import matplotlib.pyplot as plt
import numpy as np

from commodity.utils import quantile_indices, visible_ylim


# ============================================================
# line panels
# ============================================================

def lines_vary_q_across_z(
    model,
    X,
    q_probs=(0.05, 0.50, 0.80),
    z_probs=(0.05, 0.50, 0.80),
    title="",
    ylabel="",
    s_zoom=None,
    sharey=True,
):
    q_ids = quantile_indices(model.q_grid, q_probs)
    z_ids = quantile_indices(model.z_grid, z_probs)

    fig, axes = plt.subplots(
        1,
        len(z_ids),
        figsize=(5 * len(z_ids), 4),
        sharex=True,
        sharey=sharey,
    )

    if len(z_ids) == 1:
        axes = [axes]

    ylo, yhi = visible_ylim(
        model,
        X,
        q_ids=q_ids,
        z_ids=z_ids,
        s_zoom=s_zoom,
    )

    for ax, i_z, z_prob in zip(axes, z_ids, z_probs):
        for i_q, q_prob in zip(q_ids, q_probs):
            ax.plot(
                model.s_grid,
                X[:, i_q, i_z],
                label=rf"$q_{{{q_prob:.2f}}}$",
            )

        ax.set_title(rf"$z_{{{z_prob:.2f}}}$")
        ax.set_xlabel(r"storage today, $s$")

        if s_zoom is not None:
            ax.set_xlim(0.0, s_zoom)

        ax.set_ylim(ylo, yhi)
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.subplots_adjust(top=0.82)

    return fig, axes


def lines_vary_z_across_q(
    model,
    X,
    q_probs=(0.05, 0.50, 0.80),
    z_probs=(0.05, 0.50, 0.80),
    title="",
    ylabel="",
    s_zoom=None,
    sharey=True,
):
    q_ids = quantile_indices(model.q_grid, q_probs)
    z_ids = quantile_indices(model.z_grid, z_probs)

    fig, axes = plt.subplots(
        1,
        len(q_ids),
        figsize=(5 * len(q_ids), 4),
        sharex=True,
        sharey=sharey,
    )

    if len(q_ids) == 1:
        axes = [axes]

    ylo, yhi = visible_ylim(
        model,
        X,
        q_ids=q_ids,
        z_ids=z_ids,
        s_zoom=s_zoom,
    )

    for ax, i_q, q_prob in zip(axes, q_ids, q_probs):
        for i_z, z_prob in zip(z_ids, z_probs):
            ax.plot(
                model.s_grid,
                X[:, i_q, i_z],
                label=rf"$z_{{{z_prob:.2f}}}$",
            )

        ax.set_title(rf"$q_{{{q_prob:.2f}}}$")
        ax.set_xlabel(r"storage today, $s$")

        if s_zoom is not None:
            ax.set_xlim(0.0, s_zoom)

        ax.set_ylim(ylo, yhi)
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.subplots_adjust(top=0.82)

    return fig, axes


# ============================================================
# heatmaps
# ============================================================

def heatmaps_s_q_across_z(
    model,
    X,
    z_probs=(0.05, 0.50, 0.80),
    title="",
    label="",
    s_zoom=None,
    cmap="viridis",
):
    z_ids = quantile_indices(model.z_grid, z_probs)

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

    ims = []

    for ax, i_z, z_prob in zip(axes, z_ids, z_probs):
        im = ax.pcolormesh(
            S,
            Q,
            X[:, :, i_z],
            shading="auto",
            cmap=cmap,
        )

        ims.append(im)

        ax.set_title(rf"$z_{{{z_prob:.2f}}}={model.z_grid[i_z]:.4f}$")
        ax.set_xlabel(r"storage today, $s$")

        if s_zoom is not None:
            ax.set_xlim(0.0, s_zoom)

    axes[0].set_ylabel(r"inflow, $q$")

    cbar = fig.colorbar(
        ims[0],
        ax=axes,
        fraction=0.05,
        pad=0.04,
    )
    cbar.set_label(label)

    fig.suptitle(title)

    return fig, axes


def heatmaps_s_z_across_q(
    model,
    X,
    q_probs=(0.05, 0.50, 0.80),
    title="",
    label="",
    s_zoom=None,
    cmap="viridis",
):
    q_ids = quantile_indices(model.q_grid, q_probs)

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

    ims = []

    for ax, i_q, q_prob in zip(axes, q_ids, q_probs):
        im = ax.pcolormesh(
            S,
            Z,
            X[:, i_q, :],
            shading="auto",
            cmap=cmap,
        )

        ims.append(im)

        ax.set_title(rf"$q_{{{q_prob:.2f}}}={model.q_grid[i_q]:.4f}$")
        ax.set_xlabel(r"storage today, $s$")

        if s_zoom is not None:
            ax.set_xlim(0.0, s_zoom)

    axes[0].set_ylabel(r"productivity, $z$")

    cbar = fig.colorbar(
        ims[0],
        ax=axes,
        fraction=0.05,
        pad=0.04,
    )
    cbar.set_label(label)

    fig.suptitle(title)

    return fig, axes