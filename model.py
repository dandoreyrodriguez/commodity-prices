
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
        inflow_mean=0.1,
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
        self.inflow_mean = float(inflow_mean)
        self.q_grid = np.ascontiguousarray(self.inflow_mean * np.exp(self.inflow_process.grid))
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

        s_max = self.storage_max_multiple * self.inflow_mean
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
# plotting helpers
# ============================================================
# %% 
def savefig(filename, folder="figures"):
    Path(folder).mkdir(exist_ok=True)   # create folder if it doesn’t exist
    plt.tight_layout()
    plt.savefig(Path(folder) / filename, dpi=300, bbox_inches="tight")
    plt.close()

def nearest_index(grid, value):
    return int(np.argmin(np.abs(grid - value)))


def plot_storage_diagnostics(
    model,
    i_z=None,
    q_probs=(0.01, 0.20, 0.60),
    binding_tol=1e-8,
    s_zoom=0.006,
    filename="storage_diagnostics_zoom.png",
):
    if i_z is None:
        i_z = nearest_index(model.z_grid, np.median(model.z_grid))

    z_val = model.z_grid[i_z]

    binding = (model.s_policy[:, :, i_z] <= binding_tol).astype(float)
    delta = model.convenience_yield[:, :, i_z]

    S, Q = np.meshgrid(model.s_grid, model.q_grid, indexing="ij")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Binding region
    axes[0].pcolormesh(
        S,
        Q,
        binding,
        shading="auto",
        vmin=0,
        vmax=1,
    )
    axes[0].set_title(r"Binding region: $s'(s,q,z)=0$")
    axes[0].set_xlabel(r"storage today, $s$")
    axes[0].set_ylabel(r"inflow, $q$")
    axes[0].set_xlim(0.0, s_zoom)

    # 2. Convenience yield
    im1 = axes[1].pcolormesh(
        S,
        Q,
        delta,
        shading="auto",
    )
    axes[1].set_title(r"Convenience yield $\delta(s,q,z)$")
    axes[1].set_xlabel(r"storage today, $s$")
    axes[1].set_ylabel(r"inflow, $q$")
    axes[1].set_xlim(0.0, s_zoom)
    fig.colorbar(im1, ax=axes[1], label=r"$\delta$")

    # 3. Storage policy lines
    for prob in q_probs:
        q_val = np.quantile(model.q_grid, prob)
        i_q = nearest_index(model.q_grid, q_val)
        sp = model.s_policy[:, i_q, i_z]
        axes[2].plot(
            model.s_grid,
            sp,
            label=rf"$q={model.q_grid[i_q]:.4f}$",
        )

    axes[2].set_title(r"Storage policy, $s'(s,q,z)$")
    axes[2].set_xlabel(r"storage today, $s$")
    axes[2].set_ylabel(r"storage tomorrow, $s'$")
    axes[2].set_xlim(0.0, s_zoom)
    axes[2].set_ylim(-0.0005, s_zoom)
    axes[2].legend(frameon=False)

    fig.suptitle(rf"Storage diagnostics at fixed $z={z_val:.3f}$", y=1.03)

    savefig(filename)

# ============================================================
# run
# ============================================================
#%%

if __name__ == "__main__":

    model = CommodityModel(
        crra=5.0,
        beta=0.95,
        alpha=0.5,
        n_storage_states=500,
        inflow_mean=0.1,
        inflow_rho=0.85,
        inflow_sigma=0.04,
        n_inflow_states=100,
        productivity_mean=1.0,
        productivity_rho=0.90,
        productivity_sigma=0.05,
        n_productivity_states=100,
        seed=123,
    )

    # First Numba call includes compile time.
    # Second run will be much faster.
    with timer("EGM solve"):
        model.solve_egm(tol=1e-6, max_iter=500, verbose=True)

    with timer("Map to decentralised"):
        model.map_to_decentralised()

# %%

# Plots ---------------
plot_storage_diagnostics(model, s_zoom=0.03)
# %%
