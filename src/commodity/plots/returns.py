import matplotlib.pyplot as plt
import numpy as np

from commodity.pricing import expected_holding_return_fixed_z
from commodity.utils import quantile_indices


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

    surfaces = []
    stored = {}

    for i_z in z_ids:
        out = expected_holding_return_fixed_z(
            model,
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

    if s_zoom is not None:
        s_mask = model.s_grid <= s_zoom
    else:
        s_mask = np.ones_like(model.s_grid, dtype=bool)

    if shared_scale:
        visible = np.concatenate(
            [X[s_mask, :].ravel() for X in surfaces]
        )

        vmax = np.nanmax(np.abs(visible))
        vmin = -vmax
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
            local_vmax = np.nanmax(np.abs(X_plot))

            im = ax.pcolormesh(
                S_plot,
                Q_plot,
                X_plot,
                shading="auto",
                cmap=cmap,
                vmin=-local_vmax,
                vmax=local_vmax,
            )

        last_im = im

        ax.set_title(
            rf"$z_{{{z_prob:.2f}}}={model.z_grid[i_z]:.4f}$"
        )

        ax.set_xlabel(r"storage today, $s$")

    axes[0].set_ylabel(r"inflow, $q$")

    label = (
        rf"$E_t[F_{{t+1}}^{{({n-1})}}] - F_t^{{({n})}}$"
        if return_type == "level"
        else
        rf"$\left(E_t[F_{{t+1}}^{{({n-1})}}] - F_t^{{({n})}}\right)/F_t^{{({n})}}$"
    )

    fig.colorbar(
        last_im,
        ax=axes,
        shrink=0.85,
        label=label,
    )

    fig.suptitle(
        f"Expected one-period futures holding "
        f"{'payoff' if return_type == 'level' else 'return'}, n={n}",
        y=1.05,
    )

    return fig, stored


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

        out = expected_holding_return_fixed_z(
            model,
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
                label=rf"$q_{{{q_prob:.2f}}}$",
            )

        ax.axhline(0.0, linewidth=1.0, linestyle="--")

        ax.set_title(
            rf"$z_{{{z_prob:.2f}}}={model.z_grid[i_z]:.4f}$"
        )

        ax.set_xlabel(r"storage today, $s$")

        if s_zoom is not None:
            ax.set_xlim(0.0, s_zoom)

        ax.legend(frameon=False, fontsize=8)

    ylabel = (
        rf"$E_t[F_{{t+1}}^{{({n-1})}}] - F_t^{{({n})}}$"
        if return_type == "level"
        else
        r"Expected holding return"
    )

    axes[0].set_ylabel(ylabel)

    fig.suptitle(
        f"Expected one-period futures holding "
        f"{'payoff' if return_type == 'level' else 'return'}, n={n}",
        y=1.05,
    )

    return fig, stored