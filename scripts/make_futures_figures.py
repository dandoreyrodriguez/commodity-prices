import argparse
from pathlib import Path

from commodity.config import load_config
from commodity.diagnostics import storage_zoom
from commodity.io import load_model
from commodity.plots.base import make_and_save
from commodity.plots.futures import (
    expected_spot_heatmaps_s_q_across_z,
    expected_spot_heatmaps_s_z_across_q,
    futures_curves_vary_q_across_z,
    futures_curves_vary_z_across_q,
    wedge_heatmaps_s_q_across_z,
    wedge_heatmaps_s_z_across_q,
    wedge_lines_vary_q_across_z,
    wedge_term_structure_grid,
    pure_futures_term_structure_grid,
    pure_expected_spot_term_structure_grid,
)


def make_futures_figures(config_path):
    cfg = load_config(config_path)

    model_path = Path("outputs/models") / f"{cfg.name}.pkl"
    model = load_model(model_path)

    fig_cfg = cfg.figures

    q_probs = tuple(fig_cfg.q_probs)
    z_probs = tuple(fig_cfg.z_probs)
    s_probs = tuple(fig_cfg.s_probs)
    horizons = tuple(fig_cfg.horizons)

    s_zoom = storage_zoom(
        model,
        multiple=fig_cfg.s_zoom_multiple,
    )

    outdir = Path("outputs/figures") / cfg.name / "futures"

    print("SCRIPT:", Path(__file__).resolve())
    print("MODEL:", model_path.resolve())
    print("OUTDIR:", outdir.resolve())
    print("HORIZONS:", horizons)

    make_and_save(
        futures_curves_vary_q_across_z,
        "futures_grid_q_across_z.png",
        model,
        q_probs=q_probs,
        z_probs=z_probs,
        s_prob=0.75,
        T=max(horizons),
        folder=outdir,
        tight=True,
    )

    make_and_save(
        futures_curves_vary_z_across_q,
        "futures_grid_z_across_q.png",
        model,
        q_probs=q_probs,
        z_probs=z_probs,
        s_prob=0.75,
        T=max(horizons),
        folder=outdir,
        tight=True,
    )

    for h in horizons:
        make_and_save(
            wedge_heatmaps_s_q_across_z,
            f"wedge_heat_s_q_across_z_T{h}.png",
            model,
            T=h,
            z_probs=z_probs,
            s_zoom=s_zoom,
            folder=outdir,
            tight=False,
        )

        make_and_save(
            wedge_heatmaps_s_z_across_q,
            f"wedge_heat_s_z_across_q_T{h}.png",
            model,
            T=h,
            q_probs=q_probs,
            s_zoom=s_zoom,
            folder=outdir,
            tight=False,
        )

        make_and_save(
            wedge_lines_vary_q_across_z,
            f"wedge_lines_q_across_z_T{h}.png",
            model,
            T=h,
            q_probs=q_probs,
            z_probs=z_probs,
            s_zoom=s_zoom,
            folder=outdir,
            tight=True,
        )

        make_and_save(
            expected_spot_heatmaps_s_q_across_z,
            f"expected_spot_change_s_q_across_z_T{h}.png",
            model,
            T=h,
            z_probs=z_probs,
            s_zoom=s_zoom,
            change=True,
            pct=False,
            folder=outdir,
            tight=False,
        )

        make_and_save(
            expected_spot_heatmaps_s_z_across_q,
            f"expected_spot_change_s_z_across_q_T{h}.png",
            model,
            T=h,
            q_probs=q_probs,
            s_zoom=s_zoom,
            change=True,
            pct=False,
            folder=outdir,
            tight=False,
        )

    make_and_save(
        wedge_term_structure_grid,
        "wedge_term_structure_grid.png",
        model,
        s_probs=s_probs,
        q_probs=q_probs,
        z_probs=z_probs,
        T=max(horizons),
        folder=outdir,
        tight=True,
    )

    print("Making pure futures...")

    make_and_save(
        pure_futures_term_structure_grid,
        "pure_futures_term_structure_grid.png",
        model,
        s_probs=s_probs,
        q_probs=q_probs,
        z_probs=z_probs,
        T=max(horizons),
        folder=outdir,
        tight=True,
    )

    make_and_save(
        pure_expected_spot_term_structure_grid,
        "pure_expected_spot_term_structure_grid.png",
        model,
        s_probs=s_probs,
        q_probs=q_probs,
        z_probs=z_probs,
        T=max(horizons),
        folder=outdir,
        tight=True,
    )

    print(f"Saved futures figures to {outdir}")




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
    )

    args = parser.parse_args()

    make_futures_figures(args.config)


if __name__ == "__main__":
    main()