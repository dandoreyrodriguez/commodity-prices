import numpy as np
from numba import njit, prange


# ============================================================
# interpolation
# ============================================================

@njit(cache=True)
def _interp_idx_weight(grid, x):
    """Return left index and linear weight for x on an increasing grid."""
    n = grid.shape[0]

    if x <= grid[0]:
        return 0, 0.0

    if x >= grid[n - 1]:
        return n - 2, 1.0

    lo = 0
    hi = n - 1

    while hi - lo > 1:
        mid = (lo + hi) // 2
        if grid[mid] <= x:
            lo = mid
        else:
            hi = mid

    w = (x - grid[lo]) / (grid[lo + 1] - grid[lo])
    return lo, w


@njit(cache=True)
def _interp_sorted_1d(x_grid_sorted, y_sorted, x):
    """Linear interpolation on an increasing 1D grid with endpoint clipping."""
    n = x_grid_sorted.shape[0]

    if x <= x_grid_sorted[0]:
        return y_sorted[0]

    if x >= x_grid_sorted[n - 1]:
        return y_sorted[n - 1]

    lo = 0
    hi = n - 1

    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x_grid_sorted[mid] <= x:
            lo = mid
        else:
            hi = mid

    w = (x - x_grid_sorted[lo]) / (x_grid_sorted[lo + 1] - x_grid_sorted[lo])
    return (1.0 - w) * y_sorted[lo] + w * y_sorted[lo + 1]


# ============================================================
# expectations
# ============================================================

@njit(cache=True, parallel=True)
def _markov_expect_exog(X, P_q, P_z):
    """
    E[X(s, q', z') | q, z].

    Uses independence of q and z and avoids forming kron(P_q, P_z).
    X has shape (n_s, n_q, n_z).
    """
    n_s, n_q, n_z = X.shape

    tmp = np.empty((n_s, n_q, n_z))
    out = np.empty((n_s, n_q, n_z))

    # integrate over z'
    for i_s in prange(n_s):
        for j_q in range(n_q):
            for i_z in range(n_z):
                acc = 0.0
                for j_z in range(n_z):
                    acc += P_z[i_z, j_z] * X[i_s, j_q, j_z]
                tmp[i_s, j_q, i_z] = acc

    # integrate over q'
    for i_s in prange(n_s):
        for i_q in range(n_q):
            for i_z in range(n_z):
                acc = 0.0
                for j_q in range(n_q):
                    acc += P_q[i_q, j_q] * tmp[i_s, j_q, i_z]
                out[i_s, i_q, i_z] = acc

    return out


@njit(cache=True, parallel=True)
def _expect_next_policy(X, s_policy, s_grid, P_q, P_z):
    """
    E[X(s'(s,q,z), q', z') | q, z].

    Interpolates only along the storage dimension after integrating over
    exogenous next states.
    """
    EX_nodes = _markov_expect_exog(X, P_q, P_z)
    n_s, n_q, n_z = X.shape
    out = np.empty((n_s, n_q, n_z))

    for i_s in prange(n_s):
        for i_q in range(n_q):
            for i_z in range(n_z):
                idx, w = _interp_idx_weight(s_grid, s_policy[i_s, i_q, i_z])
                out[i_s, i_q, i_z] = (
                    (1.0 - w) * EX_nodes[idx, i_q, i_z]
                    + w * EX_nodes[idx + 1, i_q, i_z]
                )

    return out


@njit(cache=True, parallel=True)
def _expect_next_policy_fixed_z(X, s_policy, s_grid, P_q, P_z, i_z_fixed):
    """Same expectation as _expect_next_policy, returned only for current z=i_z_fixed."""
    EX_nodes = _markov_expect_exog(X, P_q, P_z)
    n_s, n_q, _ = X.shape
    out = np.empty((n_s, n_q))

    for i_s in prange(n_s):
        for i_q in range(n_q):
            idx, w = _interp_idx_weight(s_grid, s_policy[i_s, i_q, i_z_fixed])
            out[i_s, i_q] = (
                (1.0 - w) * EX_nodes[idx, i_q, i_z_fixed]
                + w * EX_nodes[idx + 1, i_q, i_z_fixed]
            )

    return out


@njit(cache=True, parallel=True)
def _expect_next_policy_fixed_q(X, s_policy, s_grid, P_q, P_z, i_q_fixed):
    """Same expectation as _expect_next_policy, returned only for current q=i_q_fixed."""
    EX_nodes = _markov_expect_exog(X, P_q, P_z)
    n_s, _, n_z = X.shape
    out = np.empty((n_s, n_z))

    for i_s in prange(n_s):
        for i_z in range(n_z):
            idx, w = _interp_idx_weight(s_grid, s_policy[i_s, i_q_fixed, i_z])
            out[i_s, i_z] = (
                (1.0 - w) * EX_nodes[idx, i_q_fixed, i_z]
                + w * EX_nodes[idx + 1, i_q_fixed, i_z]
            )

    return out


@njit(cache=True, parallel=True)
def _expect_next_policy_fixed_s(X, s_policy, s_grid, P_q, P_z, i_s_fixed):
    """Same expectation as _expect_next_policy, returned only for current s=i_s_fixed."""
    EX_nodes = _markov_expect_exog(X, P_q, P_z)
    _, n_q, n_z = X.shape
    out = np.empty((n_q, n_z))

    for i_q in prange(n_q):
        for i_z in range(n_z):
            idx, w = _interp_idx_weight(s_grid, s_policy[i_s_fixed, i_q, i_z])
            out[i_q, i_z] = (
                (1.0 - w) * EX_nodes[idx, i_q, i_z]
                + w * EX_nodes[idx + 1, i_q, i_z]
            )

    return out


# ============================================================
# EGM kernels
# ============================================================

@njit(cache=True, parallel=True)
def _compute_ee_values(s_grid, q_grid, z_grid, s_policy, alpha, crra):
    n_s, n_q, n_z = s_policy.shape
    EE = np.empty((n_s, n_q, n_z))

    for i_s in prange(n_s):
        s = s_grid[i_s]
        for i_q in range(n_q):
            q = q_grid[i_q]
            for i_z in range(n_z):
                z = z_grid[i_z]
                m = s + q - s_policy[i_s, i_q, i_z]
                if m < 1e-12:
                    m = 1e-12
                c = z * m ** alpha
                p = alpha * z * m ** (alpha - 1.0)
                EE[i_s, i_q, i_z] = c ** (-crra) * p

    return EE


@njit(cache=True, parallel=True)
def _egm_step(EE, s_grid, q_grid, z_grid, P_q, P_z, beta, alpha, crra):
    RHS = beta * _markov_expect_exog(EE, P_q, P_z)
    n_s, n_q, n_z = EE.shape
    s_implied = np.empty((n_s, n_q, n_z))
    theta = alpha * (1.0 - crra) - 1.0

    for i_sp in prange(n_s):
        s_prime = s_grid[i_sp]
        for i_q in range(n_q):
            q = q_grid[i_q]
            for i_z in range(n_z):
                z = z_grid[i_z]
                denom = alpha * z ** (1.0 - crra)
                m = (RHS[i_sp, i_q, i_z] / denom) ** (1.0 / theta)
                s_implied[i_sp, i_q, i_z] = m + s_prime - q

    return s_implied


@njit(cache=True, parallel=True)
def _egm_interpolate(s_implied, s_grid, q_grid):
    n_s, n_q, n_z = s_implied.shape
    out = np.empty((n_s, n_q, n_z))

    for i_q in prange(n_q):
        q = q_grid[i_q]
        for i_z in range(n_z):
            sort_idx = np.argsort(s_implied[:, i_q, i_z])
            s_impl_sorted = np.empty(n_s)
            s_prime_sorted = np.empty(n_s)

            for k in range(n_s):
                j = sort_idx[k]
                s_impl_sorted[k] = s_implied[j, i_q, i_z]
                s_prime_sorted[k] = s_grid[j]

            threshold = s_impl_sorted[0]

            for i_s in range(n_s):
                s = s_grid[i_s]
                if s <= threshold:
                    pol = 0.0
                else:
                    pol = _interp_sorted_1d(s_impl_sorted, s_prime_sorted, s)

                upper = s + q
                if pol > upper:
                    pol = upper
                if pol < 0.0:
                    pol = 0.0
                if pol > s_grid[n_s - 1]:
                    pol = s_grid[n_s - 1]

                out[i_s, i_q, i_z] = pol

    return out


# ============================================================
# decentralized objects
# ============================================================

@njit(cache=True, parallel=True)
def _compute_p_uc_m(s_grid, q_grid, z_grid, s_policy, alpha, crra):
    n_s, n_q, n_z = s_policy.shape
    m = np.empty((n_s, n_q, n_z))
    p = np.empty((n_s, n_q, n_z))
    uc = np.empty((n_s, n_q, n_z))

    for i_s in prange(n_s):
        s = s_grid[i_s]
        for i_q in range(n_q):
            q = q_grid[i_q]
            for i_z in range(n_z):
                z = z_grid[i_z]
                mm = s + q - s_policy[i_s, i_q, i_z]
                if mm < 1e-12:
                    mm = 1e-12
                c = z * mm ** alpha
                m[i_s, i_q, i_z] = mm
                p[i_s, i_q, i_z] = alpha * z * mm ** (alpha - 1.0)
                uc[i_s, i_q, i_z] = c ** (-crra)

    return m, p, uc


@njit(cache=True, parallel=True)
def _convenience_yield_kernel(p, uc, s_policy, s_grid, P_q, P_z, beta):
    A = uc * p
    EA = _expect_next_policy(A, s_policy, s_grid, P_q, P_z)
    n_s, n_q, n_z = p.shape
    delta = np.empty((n_s, n_q, n_z))

    for i_s in prange(n_s):
        for i_q in range(n_q):
            for i_z in range(n_z):
                d = p[i_s, i_q, i_z] - beta * EA[i_s, i_q, i_z] / uc[i_s, i_q, i_z]
                if d < 0.0:
                    d = 0.0
                delta[i_s, i_q, i_z] = d

    return delta


# ============================================================
# futures kernels
# ============================================================


@njit(cache=True)
def _futures_grids_at_horizon(price_s, uc, s_policy, s_grid, P_q, P_z, beta, T):
    """
    Full-grid futures objects at horizon T.

    Returns F_t^{(T)}, E_t[p_{t+T}], and wedge = F - E[p].
    """
    if T == 0:
        F0 = price_s.copy()
        Ep0 = price_s.copy()
        W0 = np.zeros_like(price_s)
        return F0, Ep0, W0

    B_prev = price_s.copy()
    D_prev = np.ones_like(price_s)
    Ep_prev = price_s.copy()

    F_grid = price_s.copy()
    Ep_new = price_s.copy()

    for _ in range(1, T + 1):
        EB = _expect_next_policy(uc * B_prev, s_policy, s_grid, P_q, P_z)
        ED = _expect_next_policy(uc * D_prev, s_policy, s_grid, P_q, P_z)
        EEp = _expect_next_policy(Ep_prev, s_policy, s_grid, P_q, P_z)

        B_new = beta * EB / uc
        D_new = beta * ED / uc
        Ep_new = EEp
        F_grid = B_new / D_new

        B_prev = B_new
        D_prev = D_new
        Ep_prev = Ep_new

    return F_grid, Ep_new, F_grid - Ep_new


@njit(cache=True)
def _futures_curve_at_state(price_s, uc, s_policy, s_grid, P_q, P_z, beta, i_s0, i_q0, i_z0, T):
    F = np.empty(T + 1)
    Ep = np.empty(T + 1)
    wedge = np.empty(T + 1)

    B_prev = price_s.copy()
    D_prev = np.ones_like(price_s)
    Ep_prev = price_s.copy()

    F[0] = price_s[i_s0, i_q0, i_z0]
    Ep[0] = price_s[i_s0, i_q0, i_z0]
    wedge[0] = 0.0

    for h in range(1, T + 1):
        EB = _expect_next_policy(uc * B_prev, s_policy, s_grid, P_q, P_z)
        ED = _expect_next_policy(uc * D_prev, s_policy, s_grid, P_q, P_z)
        EEp = _expect_next_policy(Ep_prev, s_policy, s_grid, P_q, P_z)

        B_new = beta * EB / uc
        D_new = beta * ED / uc
        Ep_new = EEp
        F_grid = B_new / D_new

        F[h] = F_grid[i_s0, i_q0, i_z0]
        Ep[h] = Ep_new[i_s0, i_q0, i_z0]
        wedge[h] = F[h] - Ep[h]

        B_prev = B_new
        D_prev = D_new
        Ep_prev = Ep_new

    return F, Ep, wedge


@njit(cache=True)
def _futures_surface_fixed_z_at_horizon(price_s, uc, s_policy, s_grid, P_q, P_z, beta, i_z_fixed, T):
    F, Ep, W = _futures_grids_at_horizon(price_s, uc, s_policy, s_grid, P_q, P_z, beta, T)
    return F[:, :, i_z_fixed].copy(), Ep[:, :, i_z_fixed].copy(), W[:, :, i_z_fixed].copy()


@njit(cache=True)
def _futures_surface_fixed_q_at_horizon(price_s, uc, s_policy, s_grid, P_q, P_z, beta, i_q_fixed, T):
    F, Ep, W = _futures_grids_at_horizon(price_s, uc, s_policy, s_grid, P_q, P_z, beta, T)
    return F[:, i_q_fixed, :].copy(), Ep[:, i_q_fixed, :].copy(), W[:, i_q_fixed, :].copy()


@njit(cache=True)
def _futures_surface_fixed_s_at_horizon(price_s, uc, s_policy, s_grid, P_q, P_z, beta, i_s_fixed, T):
    F, Ep, W = _futures_grids_at_horizon(price_s, uc, s_policy, s_grid, P_q, P_z, beta, T)
    return F[i_s_fixed, :, :].copy(), Ep[i_s_fixed, :, :].copy(), W[i_s_fixed, :, :].copy()


# ============================================================
# expected one-period holding payoff / return kernels
# ============================================================

@njit(cache=True)
def _expected_holding_grid(
    F_now_grid,
    F_next_grid,
    s_policy,
    s_grid,
    P_q,
    P_z,
):
    """
    Full-grid expected one-period holding payoff/return:

        E_t[F_{t+1}^{(n-1)}] - F_t^{(n)}
    """

    EF_next = _expect_next_policy(
        F_next_grid,
        s_policy,
        s_grid,
        P_q,
        P_z,
    )

    payoff = EF_next - F_now_grid
    ret = payoff / F_now_grid

    return EF_next, payoff, ret


@njit(cache=True)
def _expected_holding_fixed_z(
    F_now_2d,
    F_next_grid,
    s_policy,
    s_grid,
    P_q,
    P_z,
    i_z_fixed,
):
    """
    Expected holding payoff/return over (s, q),
    holding current z fixed.
    """

    EF_next = _expect_next_policy_fixed_z(
        F_next_grid,
        s_policy,
        s_grid,
        P_q,
        P_z,
        i_z_fixed,
    )

    payoff = EF_next - F_now_2d
    ret = payoff / F_now_2d

    return EF_next, payoff, ret


@njit(cache=True)
def _expected_holding_fixed_q(
    F_now_2d,
    F_next_grid,
    s_policy,
    s_grid,
    P_q,
    P_z,
    i_q_fixed,
):
    """
    Expected holding payoff/return over (s, z),
    holding current q fixed.
    """

    EF_next = _expect_next_policy_fixed_q(
        F_next_grid,
        s_policy,
        s_grid,
        P_q,
        P_z,
        i_q_fixed,
    )

    payoff = EF_next - F_now_2d
    ret = payoff / F_now_2d

    return EF_next, payoff, ret


@njit(cache=True)
def _expected_holding_fixed_s(
    F_now_2d,
    F_next_grid,
    s_policy,
    s_grid,
    P_q,
    P_z,
    i_s_fixed,
):
    """
    Expected holding payoff/return over (q, z),
    holding current s fixed.
    """

    EF_next = _expect_next_policy_fixed_s(
        F_next_grid,
        s_policy,
        s_grid,
        P_q,
        P_z,
        i_s_fixed,
    )

    payoff = EF_next - F_now_2d
    ret = payoff / F_now_2d

    return EF_next, payoff, ret
