import numpy as np


# ============================================================
# expected one-period holding payoff / return
# ============================================================

def expected_holding_return_fixed_z(
    model,
    i_z,
    n=3,
    return_type="level",
):
    """
    Computes, for fixed current z_i:

        payoff(s,q,z_i) = E_t[F_{t+1}^{(n-1)}] - F_t^{(n)}

    and optionally:

        return(s,q,z_i) = payoff / F_t^{(n)}

    Output is an array over (s, q), shape (n_s, n_q).
    """

    if n < 2:
        raise ValueError("Need n >= 2 because tomorrow maturity is n-1.")

    if return_type not in ("level", "percent"):
        raise ValueError("return_type must be either 'level' or 'percent'.")

    F_now = model.futures_surface_fixed_z(
        i_z=i_z,
        T=n,
        object_name="F",
    )

    n_s = len(model.s_grid)
    n_q = len(model.q_grid)
    n_z = len(model.z_grid)

    F_next_surfaces = [
        model.futures_surface_fixed_z(
            i_z=i_z_next,
            T=n - 1,
            object_name="F",
        )
        for i_z_next in range(n_z)
    ]

    EF_next = np.empty_like(F_now)

    for i_s in range(n_s):
        for i_q in range(n_q):

            s_next = model.s_policy[i_s, i_q, i_z]

            val = 0.0

            for i_q_next in range(n_q):
                Pq = model.P_q[i_q, i_q_next]

                for i_z_next in range(n_z):
                    Pz = model.P_z[i_z, i_z_next]

                    F_next_surface = F_next_surfaces[i_z_next]

                    F_next_interp = np.interp(
                        s_next,
                        model.s_grid,
                        F_next_surface[:, i_q_next],
                    )

                    val += Pq * Pz * F_next_interp

            EF_next[i_s, i_q] = val

    payoff = EF_next - F_now
    ret = payoff / F_now

    if return_type == "level":
        obj = payoff
    else:
        obj = ret

    return {
        "object": obj,
        "payoff": payoff,
        "return": ret,
        "F_now": F_now,
        "EF_next": EF_next,
    }