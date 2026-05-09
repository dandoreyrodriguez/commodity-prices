from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ============================================================
# model config
# ============================================================

@dataclass
class ModelConfig:

    crra: float = 5.0
    beta: float = 0.93
    alpha: float = 0.50

    n_storage_states: int = 500
    storage_max_multiple: float = 3.0
    storage_curvature: float = 3.0

    inflow_min: float = 1e-6
    inflow_scale: float = 0.04
    inflow_rho: float = 0.90
    inflow_sigma: float = 0.42
    n_inflow_states: int = 100

    productivity_mean: float = 1.0
    productivity_rho: float = 0.95
    productivity_sigma: float = 0.08
    n_productivity_states: int = 100

    seed: int | None = 123


# ============================================================
# solver config
# ============================================================

@dataclass
class SolverConfig:

    tol: float = 1e-6
    max_iter: int = 500
    verbose: bool = True


# ============================================================
# figure config
# ============================================================

@dataclass
class FigureConfig:

    q_probs: tuple = (0.01, 0.05, 0.50)
    z_probs: tuple = (0.05, 0.50, 0.80)
    s_probs: tuple = (0.01, 0.10, 0.50)

    horizons: tuple = (1, 2, 3, 6, 12)

    s_zoom_multiple: float = 1.5


# ============================================================
# full project config
# ============================================================

@dataclass
class ProjectConfig:

    name: str = "baseline"

    model: ModelConfig = field(default_factory=ModelConfig)
    solve: SolverConfig = field(default_factory=SolverConfig)
    figures: FigureConfig = field(default_factory=FigureConfig)


# ============================================================
# yaml loader
# ============================================================

def load_config(path):

    path = Path(path)

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return ProjectConfig(
        name=raw.get("name", "baseline"),

        model=ModelConfig(
            **raw.get("model", {})
        ),

        solve=SolverConfig(
            **raw.get("solve", {})
        ),

        figures=FigureConfig(
            **raw.get("figures", {})
        ),
    )