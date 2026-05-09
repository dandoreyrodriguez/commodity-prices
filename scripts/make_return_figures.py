import argparse
from pathlib import Path

from commodity.config import load_config
from commodity.diagnostics import storage_zoom
from commodity.io import load_model
from commodity.plots.base import make_and_save
from commodity.plots.returns import (
    heatmaps_expected_holding_return_s_q_across_z,
    lines_expected_holding_return_vary_s_across_z,
)


def make_return_figures(config_path):
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

    outdir = Path("outputs/figures") / cfg.name / "returns"
    outdir.mkdir(parents=True, exist_ok=True)

    for n in (2, 3, 6, 12):
        make_and_save(
            heatmaps_expected_holding_return_s_q_across_z,
            f"holding_return_heat_s_q_across_z_n{n}.png",
            model,
            z_probs=z_probs,
            n=n,
            return_type="level",
            s_zoom=s_zoom,
            folder=outdir,
            tight=False,
        )

        make_and_save(
            lines_expected_holding_return_vary_s_across_z,
            f"holding_return_lines_s_across_z_n{n}.png",
            model,
            z_probs=z_probs,
            q_probs=q_probs,
            n=n,
            return_type="level",
            s_zoom=s_zoom,
            folder=outdir,
            tight=True,
        )

    print(f"Saved expected holding return figures to {outdir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
    )

    args = parser.parse_args()

    make_return_figures(args.config)


if __name__ == "__main__":
    main()