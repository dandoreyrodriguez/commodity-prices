
######################################################
# A recursive model of commodity prices with storage #
# Fast Numba version: no 5D arrays, no full futures  #
######################################################


# %%
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numba import njit, prange


# ============================================================
# utilities
# ============================================================

# %%
@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {end - start:.4f} sec")


class Tauchen:
    def __init__(self, n, rho, sigma_eps, m=3.0, seed=None):
        self.n = int(n)
        self.rho = float(rho)
        self.sigma_eps = float(sigma_eps)
        self.m = float(m)
        self.rng = np.random.default_rng(seed)

        self.sigma_x = self.sigma_eps / np.sqrt(1.0 - self.rho**2)
        self.grid = self._build_grid()
        self.P = self._build_transition()
        self.stationary_dist = self._stationary_distribution()

    def _build_grid(self):
        x_min = -self.m * self.sigma_x
        x_max = self.m * self.sigma_x
        return np.linspace(x_min, x_max, self.n)

    def _build_transition(self):
        P = np.zeros((self.n, self.n))
        step = self.grid[1] - self.grid[0]

        for i, x in enumerate(self.grid):
            mean_next = self.rho * x

            for j, x_next in enumerate(self.grid):
                if j == 0:
                    upper = (x_next + step / 2.0 - mean_next) / self.sigma_eps
                    P[i, j] = norm.cdf(upper)
                elif j == self.n - 1:
                    lower = (x_next - step / 2.0 - mean_next) / self.sigma_eps
                    P[i, j] = 1.0 - norm.cdf(lower)
                else:
                    upper = (x_next + step / 2.0 - mean_next) / self.sigma_eps
                    lower = (x_next - step / 2.0 - mean_next) / self.sigma_eps
                    P[i, j] = norm.cdf(upper) - norm.cdf(lower)

        return P

    def _stationary_distribution(self):
        eigvals, eigvecs = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        pi = np.real(eigvecs[:, idx])
        pi = pi / pi.sum()
        pi = np.maximum(pi, 0.0)
        return pi / pi.sum()


# ============================================================
# Numba kernels
# ============================================================

@njit(cache=True)
def _interp_idx_weight(grid, x):
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

                c = z * m**alpha
                p = alpha * z * m**(alpha - 1.0)
                EE[i_s, i_q, i_z] = c**(-crra) * p

    return EE


@njit(cache=True, parallel=True)
def _markov_expect_exog(X, P_q, P_z):
    """
    E[X(s, q', z') | q, z].
    Uses independence of q and z. Does not form kron(P_q, P_z).
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
    E[X(s'(s,q,z), q', z') | q,z].
    No 5D array.
    """
    EX_nodes = _markov_expect_exog(X, P_q, P_z)
    n_s, n_q, n_z = X.shape
    out = np.empty((n_s, n_q, n_z))

    for i_s in prange(n_s):
        for i_q in range(n_q):
            for i_z in range(n_z):
                s_next = s_policy[i_s, i_q, i_z]
                idx, w = _interp_idx_weight(s_grid, s_next)
                out[i_s, i_q, i_z] = (
                    (1.0 - w) * EX_nodes[idx, i_q, i_z]
                    + w * EX_nodes[idx + 1, i_q, i_z]
                )

    return out


@njit(cache=True, parallel=True)
def _egm_step(EE, s_grid, q_grid, z_grid, P_q, P_z, beta, alpha, crra):
    RHS = beta * _markov_expect_exog(EE, P_q, P_z)

    n_s, n_q, n_z = EE.shape
    s_implied = np.empty((n_s, n_q, n_z))
    theta = alpha * (1.0 - crra) - 1.0

    for i_s in prange(n_s):
        s_prime = s_grid[i_s]
        for i_q in range(n_q):
            q = q_grid[i_q]
            for i_z in range(n_z):
                z = z_grid[i_z]
                denom = alpha * z**(1.0 - crra)
                m = (RHS[i_s, i_q, i_z] / denom)**(1.0 / theta)
                s_implied[i_s, i_q, i_z] = m + s_prime - q

    return s_implied


@njit(cache=True)
def _interp_sorted_1d(x_grid_sorted, y_sorted, x):
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

                c = z * mm**alpha

                m[i_s, i_q, i_z] = mm
                p[i_s, i_q, i_z] = alpha * z * mm**(alpha - 1.0)
                uc[i_s, i_q, i_z] = c**(-crra)

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


@njit(cache=True)
def _futures_curve_at_state(
    price_s, uc, s_policy, s_grid, P_q, P_z, beta,
    i_s0, i_q0, i_z0, T
):
    """
    One futures curve at one initial state.
    Uses 3D continuation arrays internally, but stores only T+1 output.
    """
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
def _futures_surface_fixed_z_at_horizon(
    price_s, uc, s_policy, s_grid, P_q, P_z, beta, i_z_fixed, T
):
    """
    2D map over current (s,q), holding current z fixed.
    Output shape: (n_s,n_q)
    """
    B_prev = price_s.copy()
    D_prev = np.ones_like(price_s)
    Ep_prev = price_s.copy()

    if T == 0:
        F_2d = price_s[:, :, i_z_fixed].copy()
        Ep_2d = price_s[:, :, i_z_fixed].copy()
        return F_2d, Ep_2d, np.zeros_like(F_2d)

    F_grid = price_s.copy()
    Ep_new = price_s.copy()

    for h in range(1, T + 1):
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

    F_2d = F_grid[:, :, i_z_fixed].copy()
    Ep_2d = Ep_new[:, :, i_z_fixed].copy()
    wedge_2d = F_2d - Ep_2d

    return F_2d, Ep_2d, wedge_2d


@njit(cache=True)
def _futures_surface_fixed_q_at_horizon(
    price_s, uc, s_policy, s_grid, P_q, P_z, beta, i_q_fixed, T
):
    """
    2D map over current (s,z), holding current q fixed.
    Output shape: (n_s,n_z)
    """
    B_prev = price_s.copy()
    D_prev = np.ones_like(price_s)
    Ep_prev = price_s.copy()

    if T == 0:
        F_2d = price_s[:, i_q_fixed, :].copy()
        Ep_2d = price_s[:, i_q_fixed, :].copy()
        return F_2d, Ep_2d, np.zeros_like(F_2d)

    F_grid = price_s.copy()
    Ep_new = price_s.copy()

    for h in range(1, T + 1):
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

    F_2d = F_grid[:, i_q_fixed, :].copy()
    Ep_2d = Ep_new[:, i_q_fixed, :].copy()
    wedge_2d = F_2d - Ep_2d

    return F_2d, Ep_2d, wedge_2d


@njit(cache=True)
def _futures_surface_fixed_s_at_horizon(
    price_s, uc, s_policy, s_grid, P_q, P_z, beta, i_s_fixed, T
):
    """
    2D map over current (q,z), holding current s fixed.
    Output shape: (n_q,n_z)
    """
    B_prev = price_s.copy()
    D_prev = np.ones_like(price_s)
    Ep_prev = price_s.copy()

    if T == 0:
        F_2d = price_s[i_s_fixed, :, :].copy()
        Ep_2d = price_s[i_s_fixed, :, :].copy()
        return F_2d, Ep_2d, np.zeros_like(F_2d)

    F_grid = price_s.copy()
    Ep_new = price_s.copy()

    for h in range(1, T + 1):
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

    F_2d = F_grid[i_s_fixed, :, :].copy()
    Ep_2d = Ep_new[i_s_fixed, :, :].copy()
    wedge_2d = F_2d - Ep_2d

    return F_2d, Ep_2d, wedge_2d


def _select_object(F, Ep, W, object_name):
    if object_name == "F":
        return F
    elif object_name == "Ep":
        return Ep
    elif object_name == "wedge":
        return W
    else:
        raise ValueError("object_name must be 'F', 'Ep', or 'wedge'")


# ============================================================
# model
# ============================================================

class CommodityModel:
    def __init__(
        self,
        crra,
        beta,
        alpha,
        n_storage_states=100,
        storage_max_multiple=4.0,
        storage_curvature=3.0,
        inflow_min=0.002,
        inflow_rho=0.9,
        inflow_sigma=0.1,
        n_inflow_states=20,
        productivity_mean=1.0,
        productivity_rho=0.9,
        productivity_sigma=0.05,
        n_productivity_states=20,
        seed=None,
    ):
        self.crra = float(crra)
        self.beta = float(beta)
        self.alpha = float(alpha)

        seed_q = seed
        seed_z = None if seed is None else seed + 1

        self.inflow_process = Tauchen(
            n=n_inflow_states,
            rho=inflow_rho,
            sigma_eps=inflow_sigma,
            m=2.0,
            seed=seed_q,
        )
        self.inflow_min = float(inflow_min)
        self.q_grid = self.inflow_min + np.ascontiguousarray(np.exp(self.inflow_process.grid))
        self.P_q = np.ascontiguousarray(self.inflow_process.P)
        self.stationary_dist_q = self.inflow_process.stationary_dist
        self.n_q = int(n_inflow_states)

        self.productivity_process = Tauchen(
            n=n_productivity_states,
            rho=productivity_rho,
            sigma_eps=productivity_sigma,
            m=2.0,
            seed=seed_z,
        )
        self.productivity_mean = float(productivity_mean)
        self.z_grid = np.ascontiguousarray(self.productivity_mean * np.exp(self.productivity_process.grid))
        self.P_z = np.ascontiguousarray(self.productivity_process.P)
        self.stationary_dist_z = self.productivity_process.stationary_dist
        self.n_z = int(n_productivity_states)

        self.n_s = int(n_storage_states)
        self.storage_max_multiple = float(storage_max_multiple)
        self.storage_curvature = float(storage_curvature)

        s_max = self.storage_max_multiple * float(self.q_grid @ self.stationary_dist_q)
        self.s_grid = np.ascontiguousarray(self.make_storage_grid(s_max, self.n_s, self.storage_curvature))

        shape = (self.n_s, self.n_q, self.n_z)
        self.s_policy = np.zeros(shape)
        self.m_policy = np.zeros(shape)
        self.price_s = np.zeros(shape)
        self.u_c = np.zeros(shape)
        self.convenience_yield = np.zeros(shape)
        self.expected_Mp_next = np.zeros(shape)

    @staticmethod
    def make_storage_grid(s_max, n_s, curvature):
        x = np.linspace(0.0, 1.0, n_s)
        return s_max * x**curvature

    def solve_egm(self, tol=1e-6, max_iter=500, verbose=True):
        s = self.s_grid[:, None, None]
        q = self.q_grid[None, :, None]

        # IMPORTANT: broadcast initial guess to full (s,q,z).
        # Without this, the model silently has z dimension 1.
        s_policy0 = np.clip(0.5 * (s + q), 0.0, self.s_grid[-1])
        s_policy = np.ascontiguousarray(
            np.broadcast_to(s_policy0, (self.n_s, self.n_q, self.n_z)).copy()
        )

        error = 1.0
        iteration = 0

        while error > tol and iteration < max_iter:
            EE = _compute_ee_values(
                self.s_grid, self.q_grid, self.z_grid,
                s_policy, self.alpha, self.crra
            )

            s_implied = _egm_step(
                EE,
                self.s_grid,
                self.q_grid,
                self.z_grid,
                self.P_q,
                self.P_z,
                self.beta,
                self.alpha,
                self.crra,
            )

            s_policy_new = _egm_interpolate(s_implied, self.s_grid, self.q_grid)

            error = float(np.max(np.abs(s_policy_new - s_policy)))
            s_policy = np.ascontiguousarray(s_policy_new)
            iteration += 1

            if verbose and iteration % 25 == 0:
                print(f"Iteration {iteration}, policy error = {error:.3e}")

        if verbose:
            print(f"Stopped after {iteration} iterations, error = {error:.3e}")

        self.s_policy = s_policy
        self.m_policy = np.maximum(
            self.s_grid[:, None, None] + self.q_grid[None, :, None] - self.s_policy,
            1e-12,
        )

        return self

    def map_to_decentralised(self):
        self.m_policy, self.price_s, self.u_c = _compute_p_uc_m(
            self.s_grid,
            self.q_grid,
            self.z_grid,
            self.s_policy,
            self.alpha,
            self.crra,
        )

        self.convenience_yield = _convenience_yield_kernel(
            self.price_s,
            self.u_c,
            self.s_policy,
            self.s_grid,
            self.P_q,
            self.P_z,
            self.beta,
        )

        self.expected_Mp_next = self.price_s - self.convenience_yield

        return self

    def futures_curve_at_index(self, i_s, i_q, i_z, T=12):
        F, Ep, wedge = _futures_curve_at_state(
            self.price_s,
            self.u_c,
            self.s_policy,
            self.s_grid,
            self.P_q,
            self.P_z,
            self.beta,
            int(i_s),
            int(i_q),
            int(i_z),
            int(T),
        )

        return {
            "maturity": np.arange(T + 1),
            "F": F,
            "Ep": Ep,
            "wedge": wedge,
        }

    def futures_curve_at_values(self, s, q, z, T=12):
        i_s = nearest_index(self.s_grid, s)
        i_q = nearest_index(self.q_grid, q)
        i_z = nearest_index(self.z_grid, z)

        out = self.futures_curve_at_index(i_s, i_q, i_z, T=T)
        out.update({
            "i_s": i_s,
            "i_q": i_q,
            "i_z": i_z,
            "s": self.s_grid[i_s],
            "q": self.q_grid[i_q],
            "z": self.z_grid[i_z],
        })
        return out

    def futures_surface_fixed_z(self, i_z, T=12, object_name="wedge"):
        F, Ep, W = _futures_surface_fixed_z_at_horizon(
            self.price_s,
            self.u_c,
            self.s_policy,
            self.s_grid,
            self.P_q,
            self.P_z,
            self.beta,
            int(i_z),
            int(T),
        )
        return _select_object(F, Ep, W, object_name)

    def futures_surface_fixed_q(self, i_q, T=12, object_name="wedge"):
        F, Ep, W = _futures_surface_fixed_q_at_horizon(
            self.price_s,
            self.u_c,
            self.s_policy,
            self.s_grid,
            self.P_q,
            self.P_z,
            self.beta,
            int(i_q),
            int(T),
        )
        return _select_object(F, Ep, W, object_name)

    def futures_surface_fixed_s(self, i_s, T=12, object_name="wedge"):
        F, Ep, W = _futures_surface_fixed_s_at_horizon(
            self.price_s,
            self.u_c,
            self.s_policy,
            self.s_grid,
            self.P_q,
            self.P_z,
            self.beta,
            int(i_s),
            int(T),
        )
        return _select_object(F, Ep, W, object_name)

# ============================================================
# run model
# ============================================================

if __name__ == "__main__":

    model = CommodityModel(
        crra=10.0,
        beta=0.95,
        alpha=0.5,
        n_storage_states=500,
        inflow_min=0.01,
        inflow_rho=0.80,
        inflow_sigma=0.2,
        n_inflow_states=150,
        productivity_mean=1.0,
        productivity_rho=0.90,
        productivity_sigma=0.10,
        n_productivity_states=150,
        seed=123,
    )

    with timer("EGM solve"):
        model.solve_egm(
            tol=1e-6,
            max_iter=500,
            verbose=True,
        )

    with timer("Map to decentralised"):
        model.map_to_decentralised()

    print("Model solved and mapped.")


# %% 
# ============================================================
# plotting + diagnostics: return figures, do not auto-close
# ============================================================


def nearest_index(grid, value):
    return int(np.argmin(np.abs(grid - value)))


def quantile_indices(grid, probs):
    return [nearest_index(grid, np.quantile(grid, p)) for p in probs]


def save_fig(fig, filename, folder="figures", tight=True):
    Path(folder).mkdir(exist_ok=True)

    if tight:
        fig.tight_layout()

    fig.savefig(Path(folder) / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_and_save(fig_func, filename, *args, tight=True, **kwargs):
    """
    Create figure → save → close immediately.
    Use tight=False for figures using constrained_layout=True.
    """
    out = fig_func(*args, **kwargs)
    fig = out[0] if isinstance(out, tuple) else out
    save_fig(fig, filename, tight=tight)

def visible_ylim(model, X, q_ids=None, z_ids=None, s_zoom=None, pad=0.06):
    """
    Compute y-limits using only the part of the curves visible under s_zoom.
    Works for shared-y panel plots.
    """
    s_mask = model.s_grid <= s_zoom if s_zoom is not None else np.ones_like(model.s_grid, dtype=bool)

    vals = []

    if q_ids is not None and z_ids is not None:
        for i_q in q_ids:
            for i_z in z_ids:
                vals.append(X[s_mask, i_q, i_z])

    vals = np.concatenate(vals)

    ymin = np.nanmin(vals)
    ymax = np.nanmax(vals)

    gap = pad * (ymax - ymin if ymax > ymin else 1.0)

    return max(0.0, ymin - gap), ymax + gap

def consumption(model):
    return model.z_grid[None, None, :] * model.m_policy ** model.alpha


def binding_region(model, tol=1e-8):
    return (model.s_policy <= tol).astype(float)


# ============================================================
# LINE PANELS
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
        ax.set_xlabel(r"$s$")

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
        ax.set_xlabel(r"$s$")

        if s_zoom is not None:
            ax.set_xlim(0.0, s_zoom)

        ax.set_ylim(ylo, yhi)
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.subplots_adjust(top=0.82)

    return fig, axes

# ============================================================
# heatmaps with constrained_layout=True
# ============================================================

def heatmaps_s_q_across_z(
    model,
    X,
    z_probs=(0.05, 0.50, 0.80),
    title="",
    label="",
    s_zoom=None,
    shared_scale=True,
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

    # global scale if requested
    if shared_scale:
        vmin = np.min(X)
        vmax = np.max(X)
    else:
        vmin = vmax = None

    ims = []

    for ax, i_z, z_prob in zip(axes, z_ids, z_probs):
        im = ax.pcolormesh(
            S,
            Q,
            X[:, :, i_z],
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ims.append(im)

        ax.set_title(rf"$z_{{{z_prob:.2f}}}={model.z_grid[i_z]:.4f}$")
        ax.set_xlabel(r"storage today, $s$")

        if s_zoom is not None:
            ax.set_xlim(0.0, s_zoom)

    axes[0].set_ylabel(r"inflow, $q$")

    # colorbars
    if shared_scale:
        cbar = fig.colorbar(ims[0], ax=axes, fraction=0.05, pad=0.04)
        cbar.set_label(label)
    else:
        for ax, im in zip(axes, ims):
            cbar = fig.colorbar(im, ax=ax)
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
    shared_scale=True,
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

    # global scale
    if shared_scale:
        vmin = np.min(X)
        vmax = np.max(X)
    else:
        vmin = vmax = None

    ims = []

    for ax, i_q, q_prob in zip(axes, q_ids, q_probs):
        im = ax.pcolormesh(
            S,
            Z,
            X[:, i_q, :],
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ims.append(im)

        ax.set_title(rf"$q_{{{q_prob:.2f}}}={model.q_grid[i_q]:.4f}$")
        ax.set_xlabel(r"storage today, $s$")

        if s_zoom is not None:
            ax.set_xlim(0.0, s_zoom)

    axes[0].set_ylabel(r"productivity, $z$")

    # colorbars
    if shared_scale:
        cbar = fig.colorbar(ims[0], ax=axes, fraction=0.05, pad=0.04)
        cbar.set_label(label)
    else:
        for ax, im in zip(axes, ims):
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(label)

    fig.suptitle(title)

    return fig, axes

def futures_curves_vary_q_across_z(
    model,
    q_probs=(0.05, 0.50, 0.80),
    z_probs=(0.05, 0.50, 0.80),
    i_s=None,
    T=12,
):
    if i_s is None:
        i_s = nearest_index(model.s_grid, np.median(model.s_grid))

    q_ids = quantile_indices(model.q_grid, q_probs)
    z_ids = quantile_indices(model.z_grid, z_probs)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(q_ids)))

    fig, axes = plt.subplots(
        1,
        len(z_ids),
        figsize=(5 * len(z_ids), 4),
        sharey=True,
    )

    if len(z_ids) == 1:
        axes = [axes]

    stored = {}

    for ax, i_z, z_prob in zip(axes, z_ids, z_probs):
        stored[i_z] = {}

        for color, i_q, q_prob in zip(colors, q_ids, q_probs):
            out = model.futures_curve_at_index(i_s, i_q, i_z, T=T)
            stored[i_z][i_q] = out

            ax.plot(
                out["maturity"],
                out["F"],
                color=color,
                marker="o",
                label=rf"$F$, $q_{{{q_prob:.2f}}}$",
            )

            ax.plot(
                out["maturity"],
                out["Ep"],
                color=color,
                linestyle="--",
                marker="s",
                label=rf"$E[p]$, $q_{{{q_prob:.2f}}}$",
            )

        ax.set_title(rf"$z_{{{z_prob:.2f}}}$")
        ax.set_xlabel("maturity")
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel("price")
    fig.suptitle(
        rf"Futures vs expected spot, fixed $s={model.s_grid[i_s]:.4f}$"
    )
    fig.subplots_adjust(top=0.82)

    return fig, axes, stored

def futures_curves_vary_z_across_q(
    model,
    q_probs=(0.05, 0.50, 0.80),
    z_probs=(0.05, 0.50, 0.80),
    i_s=None,
    T=12,
):
    if i_s is None:
        i_s = nearest_index(model.s_grid, np.median(model.s_grid))

    q_ids = quantile_indices(model.q_grid, q_probs)
    z_ids = quantile_indices(model.z_grid, z_probs)

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(z_ids)))

    fig, axes = plt.subplots(
        1,
        len(q_ids),
        figsize=(5 * len(q_ids), 4),
        sharey=True,
    )

    if len(q_ids) == 1:
        axes = [axes]

    stored = {}

    for ax, i_q, q_prob in zip(axes, q_ids, q_probs):
        stored[i_q] = {}

        for color, i_z, z_prob in zip(colors, z_ids, z_probs):
            out = model.futures_curve_at_index(i_s, i_q, i_z, T=T)
            stored[i_q][i_z] = out

            ax.plot(
                out["maturity"],
                out["F"],
                color=color,
                marker="o",
                label=rf"$F$, $z_{{{z_prob:.2f}}}$",
            )

            ax.plot(
                out["maturity"],
                out["Ep"],
                color=color,
                linestyle="--",
                marker="s",
                label=rf"$E[p]$, $z_{{{z_prob:.2f}}}$",
            )

        ax.set_title(rf"$q_{{{q_prob:.2f}}}$")
        ax.set_xlabel("maturity")
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel("price")
    fig.suptitle(
        rf"Futures vs expected spot, fixed $s={model.s_grid[i_s]:.4f}$"
    )
    fig.subplots_adjust(top=0.82)

    return fig, axes, stored


# ============================================================
# RUN DIAGNOSTICS
# ============================================================

q_probs = (0.05, 0.50, 0.80)
z_probs = (0.05, 0.50, 0.80)
s_zoom = 0.10

C = consumption(model)
B = binding_region(model)

objects = {
    "s_policy": (model.s_policy, r"Storage policy $s'(s,q,z)$", r"$s'$"),
    "m_policy": (model.m_policy, r"Use $m(s,q,z)$", r"$m$"),
    "c_policy": (C, r"Consumption $c(s,q,z)$", r"$c$"),
    "price": (model.price_s, r"Spot price $p^s(s,q,z)$", r"$p^s$"),
    "delta": (model.convenience_yield, r"Convenience yield $\delta(s,q,z)$", r"$\delta$"),
    "binding": (B, r"Binding region $s'(s,q,z)=0$", "binding"),
}

# ---------- line panels: normal tight layout ----------
for name, (X, title, ylabel) in objects.items():

    make_and_save(
        lines_vary_q_across_z,
        f"{name}_lines_q_across_z.png",
        model,
        X,
        q_probs=q_probs,
        z_probs=z_probs,
        title=title + r": varying $q$ across $z$",
        ylabel=ylabel,
        s_zoom=s_zoom,
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
        ylabel=ylabel,
        s_zoom=s_zoom,
        tight=True,
    )

# ---------- heatmap panels: NO tight layout ----------
for name, (X, title, label) in objects.items():

    make_and_save(
        heatmaps_s_q_across_z,
        f"{name}_heat_s_q_across_z.png",
        model,
        X,
        z_probs=z_probs,
        title=title + r": $(s,q)$ across $z$",
        label=label,
        s_zoom=s_zoom,
        tight=False,   # important
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
        tight=False,   # important
    )

# ---------- futures ----------
make_and_save(
    futures_curves_vary_q_across_z,
    "futures_grid_q_across_z.png",
    model,
    q_probs=q_probs,
    z_probs=z_probs,
    T=12,
    tight=True,
)

make_and_save(
    futures_curves_vary_z_across_q,
    "futures_grid_z_across_q.png",
    model,
    q_probs=q_probs,
    z_probs=z_probs,
    T=12,
    tight=True,
)