import numpy as np

from commodity.utils import nearest_index, quantile_indices


# ============================================================
# basic derived objects
# ============================================================

def consumption(model):
    return model.z_grid[None, None, :] * model.m_policy**model.alpha


def binding_region(model, tol=1e-8):
    return (model.s_policy <= tol).astype(float)


def storage_zoom(model, multiple=1.5):
    return multiple * float(model.q_grid @ model.stationary_dist_q)


def core_objects(model):
    C = consumption(model)
    B = binding_region(model)

    return {
        "s_policy": (
            model.s_policy,
            r"Storage policy $s'(s,q,z)$",
            r"$s'$",
        ),
        "m_policy": (
            model.m_policy,
            r"Use $m(s,q,z)$",
            r"$m$",
        ),
        "c_policy": (
            C,
            r"Consumption $c(s,q,z)$",
            r"$c$",
        ),
        "price": (
            model.price_s,
            r"Spot price $p^s(s,q,z)$",
            r"$p^s$",
        ),
        "delta": (
            model.convenience_yield,
            r"Convenience yield $\delta(s,q,z)$",
            r"$\delta$",
        ),
        "binding": (
            B,
            r"Binding region $s'(s,q,z)=0$",
            "binding",
        ),
    }


# ============================================================
# futures / expected spot diagnostics
# ============================================================

def futures_vs_expected_spot_slope_grid(
    model,
    s_probs=(0.01, 0.10, 0.50),
    q_probs=(0.01, 0.05, 0.50),
    z_probs=(0.05, 0.50, 0.80),
    T=12,
):
    """
    Returns a list of dictionaries comparing:

        1. expected spot slope: E[p_{t+T}] - p_t
        2. futures slope: F_{t,t+T} - p_t
        3. wedge: F - E[p]
    """

    rows = []

    for s_prob in s_probs:
        i_s = nearest_index(
            model.s_grid,
            np.quantile(model.s_grid, s_prob),
        )

        for q_prob in q_probs:
            i_q = nearest_index(
                model.q_grid,
                np.quantile(model.q_grid, q_prob),
            )

            for z_prob in z_probs:
                i_z = nearest_index(
                    model.z_grid,
                    np.quantile(model.z_grid, z_prob),
                )

                out = model.futures_curve_at_index(
                    i_s,
                    i_q,
                    i_z,
                    T=T,
                )

                p0 = out["Ep"][0]

                rows.append(
                    {
                        "s_prob": s_prob,
                        "q_prob": q_prob,
                        "z_prob": z_prob,
                        "s": model.s_grid[i_s],
                        "q": model.q_grid[i_q],
                        "z": model.z_grid[i_z],
                        "p0": p0,
                        "Ep_T": out["Ep"][T],
                        "F_T": out["F"][T],
                        "expected_spot_slope": out["Ep"][T] - p0,
                        "futures_slope": out["F"][T] - p0,
                        "wedge": out["wedge"][T],
                    }
                )

    return rows


# ============================================================
# expected spot surfaces
# ============================================================

def expected_spot_surface_fixed_z(model, i_z, T=12):
    return model.futures_surface_fixed_z(
        i_z,
        T=T,
        object_name="Ep",
    )


def expected_spot_surface_fixed_q(model, i_q, T=12):
    return model.futures_surface_fixed_q(
        i_q,
        T=T,
        object_name="Ep",
    )


def expected_spot_change_fixed_z(model, i_z, T=12, pct=False):
    Ep = expected_spot_surface_fixed_z(model, i_z, T=T)
    p0 = model.price_s[:, :, i_z]

    out = Ep - p0

    if pct:
        out = out / p0

    return out


def expected_spot_change_fixed_q(model, i_q, T=12, pct=False):
    Ep = expected_spot_surface_fixed_q(model, i_q, T=T)
    p0 = model.price_s[:, i_q, :]

    out = Ep - p0

    if pct:
        out = out / p0

    return out


# ============================================================
# wedge surfaces
# ============================================================

def wedge_surface_fixed_z(model, i_z, T=12):
    return model.futures_surface_fixed_z(
        i_z,
        T=T,
        object_name="wedge",
    )


def wedge_surface_fixed_q(model, i_q, T=12):
    return model.futures_surface_fixed_q(
        i_q,
        T=T,
        object_name="wedge",
    )


def wedge_surface_fixed_s(model, i_s, T=12):
    return model.futures_surface_fixed_s(
        i_s,
        T=T,
        object_name="wedge",
    )