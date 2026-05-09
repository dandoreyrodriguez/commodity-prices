import argparse
from pathlib import Path

from commodity.config import load_config
from commodity.diagnostics import core_objects, storage_zoom
from commodity.io import load_model
from commodity.plots.base import make_and_save
from commodity.plots.core import (
    heatmaps_s_q_across_z,
    heatmaps_s_z_across_q,
    lines_vary_q_across_z,
    lines_vary_z_across_q,
)


def make_core_figures(config_path):
    cfg = load_config(config_path)

    model_path = Path("outputs/models") / f"{cfg.name}.pkl"
    model = load_model(model_path)

    fig_cfg = cfg.figures

    q_probs = tuple(fig_cfg.q_probs)
    z_probs = tuple(fig_cfg.z_probs)

    s_zoom = storage_zoom(
        model,
        multiple=fig_cfg.s_zoom_multiple,
    )

    outdir = Path("outputs/figures") / cfg.name / "core"

    objects = core_objects(model)

    for name, (X, title, label) in objects.items():

        make_and_save(
            lines_vary_q_across_z,
            f"{name}_lines_q_across_z.png",
            model,
            X,
            q_probs=q_probs,
            z_probs=z_probs,
            title=title + r": varying $q$ across $z$",
            ylabel=label,
            s_zoom=s_zoom,
            folder=outdir,
            tight=True,
        )

        make_and_save(
            lines_vary_z_across_q,
            f"{name}_lines_z_across_q.png",
            model,
            X,
            q_probs=q_probs,
            z_probs=z_probs,
            title=title + r": varying $z$ across $q$",
            ylabel=label,
            s_zoom=s_zoom,
            folder=outdir,
            tight=True,
        )

        make_and_save(
            heatmaps_s_q_across_z,
            f"{name}_heat_s_q_across_z.png",
            model,
            X,
            z_probs=z_probs,
            title=title + r": $(s,q)$ across $z$",
            label=label,
            s_zoom=s_zoom,
            folder=outdir,
            tight=False,
        )

        make_and_save(
            heatmaps_s_z_across_q,
            f"{name}_heat_s_z_across_q.png",
            model,
            X,
            q_probs=q_probs,
            title=title + r": $(s,z)$ across $q$",
            label=label,
            s_zoom=s_zoom,
            folder=outdir,
            tight=False,
        )

    print(f"Saved core figures to {outdir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
    )

    args = parser.parse_args()

    make_core_figures(args.config)


if __name__ == "__main__":
    main()