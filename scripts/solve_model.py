import argparse
from pathlib import Path

from commodity.config import load_config
from commodity.io import save_model
from commodity.model import CommodityModel
from commodity.utils import timer


def solve_from_config(config_path):
    cfg = load_config(config_path)

    model = CommodityModel(**cfg.model.__dict__)

    with timer("EGM solve"):
        model.solve_egm(**cfg.solve.__dict__)

    with timer("Map to decentralised"):
        model.map_to_decentralised()

    out_path = Path("outputs/models") / f"{cfg.name}.pkl"
    save_model(model, out_path)

    print(f"Saved solved model to {out_path}")

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
    )

    args = parser.parse_args()

    solve_from_config(args.config)


if __name__ == "__main__":
    main()