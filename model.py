######################################################
# A recursive model of commodity prices with storage #
######################################################


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import comb
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

# 0. Rouwenhorst method for approximating AR(1) processes ----------------

# %%


class Rouwenhorst:
    def __init__(self, n, rho, sigma_eps, seed=None):
        """
        Approximate AR(1):
            z_{t+1} = rho z_t + sigma_eps eps_{t+1}

        Parameters
        ----------
        n : int
            Number of grid points.
        rho : float
            AR(1) persistence.
        sigma_eps : float
            Innovation standard deviation.
        seed : int or None
            Random seed.
        """
        self.m = n - 1  # number of binary flips
        self.rho = rho
        self.sigma_eps = sigma_eps
        self.rng = np.random.default_rng(seed)

        self.p = (1 + rho) / 2  # probability of staying in the same state
        self.sigma_z = sigma_eps / np.sqrt(1 - rho**2)

        self.P = self._build_transition()
        self.grid = self._build_grid()
        self.stationary_dist = self._stationary_distribution()

    def _build_transition(self):
        """Build the transition matrix using Rouwenhorst's method."""
        m = self.m
        p = self.p

        # Initialise with the single binary transition matrix
        P = np.array([[p, 1 - p], [1 - p, p]])

        # iterate
        for j in range(2, m + 1):
            # create transition matrix, must be j+1 by j+1
            P_new = np.zeros((j + 1, j + 1))
            # top left
            P_new[:-1, :-1] += p * P
            # top right
            P_new[:-1, 1:] += (1 - p) * P
            # bottom left
            P_new[1:, :-1] += (1 - p) * P
            # bottom right
            P_new[1:, 1:] += p * P
            # normalise
            P_new[1:-1, :] /= 2
            P = P_new
        # return P
        return P

    def _build_grid(self):
        raw_grid = np.linspace(-self.m, self.m, self.m + 1)
        z = self.sigma_z * raw_grid / np.sqrt(self.m)
        return z

    def _stationary_distribution(self):
        pi = np.array([comb(self.m, k) * (0.5**self.m) for k in range(self.m + 1)])
        return pi

    def simulate(self, T, start_idx=None):
        idx = np.empty(T, dtype=int)

        if start_idx is None:
            idx[0] = self.rng.choice(self.m + 1, p=self.stationary_dist)
        else:
            idx[0] = start_idx

        for t in range(1, T):
            idx[t] = self.rng.choice(self.m + 1, p=self.P[idx[t - 1]])

        z_path = self.grid[idx]
        return z_path, idx

    def simulate_ar1(self, T, start=0.0):
        z = np.empty(T)
        z[0] = start

        for t in range(1, T):
            eps = self.rng.normal()
            z[t] = self.rho * z[t - 1] + self.sigma_eps * eps

        return z


# 1. The model ------------------------------------------------------------------


class CommodityModel:

    def __init__(
        self,
        crra,
        beta,
        alpha,
        n_storage_states=50,
        storage_max_multiple=3.0,
        storage_curvature=3.0,
        inflow_mean=0.1,
        inflow_rho=0.9,
        inflow_sigma=0.1,
        n_inflow_states=20,
        seed=None,
    ):

        # consumer preferences
        self.crra = crra
        self.beta = beta

        # firm technology
        self.alpha = alpha

        # Rouwenhorst approximation for inflow process (note this is for x, where x = log(q) - log(mu), so mean is zero)
        self.inflow_process = Rouwenhorst(
            n=n_inflow_states, rho=inflow_rho, sigma_eps=inflow_sigma, seed=seed
        )
        self.inflow_mean = inflow_mean
        self.q_grid = inflow_mean * np.exp(self.inflow_process.grid)
        self.P_q = self.inflow_process.P
        self.stationary_dist_q = self.inflow_process.stationary_dist
        self.n_q = n_inflow_states

        # set up storage grid
        self.n_s = n_storage_states
        self.storage_max_multiple = storage_max_multiple
        s_max = storage_max_multiple * inflow_mean
        self.s_grid = self.make_storage_grid(s_max, n_storage_states, storage_curvature)
        self.storage_curvature = storage_curvature
        # initialise value function
        self.V = np.zeros((self.n_s, self.n_q))
        # initialise policy function
        self.s_policy = np.zeros((self.n_s, self.n_q))
        # initialise commodity employed for production
        self.m_policy = np.zeros((self.n_s, self.n_q))
        # initialise the convenience yield
        self.convenience_yield = np.zeros((self.n_s, self.n_q))

    def make_storage_grid(self, s_max, n_s, curvature):
        x = np.linspace(0.0, 1.0, n_s)
        return s_max * x**curvature

    def utility(self, c):
        if self.crra == 1:
            return np.log(c)
        else:
            return (c ** (1 - self.crra) - 1) / (1 - self.crra)

    def production(self, s):
        return s**self.alpha

    def marginal_utility(self, c):
        return c ** (-self.crra)

    def marginal_product(self, s):
        return self.alpha * s ** (self.alpha - 1)

    def V_0(self):
        # use c = F(mean_inflow) as initial guess for consumption in value function
        c = self.production(self.inflow_mean)
        u = self.utility(c) / (1 - self.beta)
        V = np.full((self.n_s, self.n_q), u)
        return V

    def interp_weights(self, x):
        """
        Find x on self.s_grid and return idx, w such that:
            f(x) = (1-w) f(s_grid[idx]) + w f(s_grid[idx+1])
        """
        # prep grid
        x = np.asarray(x)
        x = np.clip(x, self.s_grid[0], self.grid[-1])
        # prep index
        idx = np.searchsorted(self.s_grid, x) - 1
        idx = np.clip(x, 0, self.n_s - 2)
        # the two bounds
        s_0 = self.s_grid[idx]
        s_1 = self.s_grid[idx + 1]
        w = (x - s_0) / (s_1 - s_0)
        return idx, w

    def interp_1d_on_s(self, y, x):
        """
        Interpolate y(s) at x.
        """
        idx, w = self.interp_weights(x)
        return (1 - w) * y[idx] + w * y[idx + 1]

    def interp_2d_on_s(self, X, x):
        """ "
        Interpolate X(s,q) along s dimension at x. X is shape (n_s, n_q).

        """
        idx, w = self.interp_weights(x)
        X_low = X[idx, :]
        X_high = X[idx + 1, :]
        return (1 - w[..., None]) * X_low + w[..., None] * X_high

    def eval_next_by_q(self, X):
        """
        Given X(s,q) evaluate X(s'(s,q),q') for all q'.
        """
        return self.interp_2d_on_s(X, self.s_policy)

    def bellman_operator(self, V_old):
        V_new = np.zeros_like(V_old)
        s_policy = np.zeros_like(self.s_policy)

        # precompute expected continuation value on the storage grid
        EV_on_grid = V_old @ self.P_q.T  # shape(n_s, n_q)

        # interpolate E[V(s', q' = j) | q] as a function of s'
        EV_interp = [
            interp1d(
                self.s_grid,
                EV_on_grid[:, i_q],
                kind="linear",
                fill_value="extrapolate",
            )
            for i_q in range(self.n_q)
        ]

        for i_s, s in enumerate(self.s_grid):
            for i_q, q in enumerate(self.q_grid):
                upper = min(s + q, self.s_grid[-1])

                def objective(s_next):
                    m = s + q - s_next
                    if m <= 0:
                        return 1e10

                    c = self.production(m)
                    u = self.utility(c)
                    EV = EV_interp[i_q](s_next)

                    return -(u + self.beta * EV)

                result = minimize_scalar(
                    objective,
                    bounds=(0.0, upper),
                    method="bounded",
                    options={"xatol": 1e-5},
                )

                V_new[i_s, i_q] = -result.fun
                s_policy[i_s, i_q] = result.x

        return V_new, s_policy

    def policy_evaluation_step(self, V_old, s_policy):

        V_new = np.zeros_like(V_old)

        EV_on_grid = V_old @ self.P_q.T

        EV_interp = [
            interp1d(
                self.s_grid,
                EV_on_grid[:, i_q],
                kind="linear",
                fill_value="extrapolate",
            )
            for i_q in range(self.n_q)
        ]

        for i_s, s in enumerate(self.s_grid):
            for i_q, q in enumerate(self.q_grid):
                s_next = s_policy[i_s, i_q]
                m = s + q - s_next

                if m <= 0:
                    V_new[i_s, i_q] = -1e10
                else:
                    c = self.production(m)
                    V_new[i_s, i_q] = self.utility(c) + self.beta * EV_interp[i_q](
                        s_next
                    )

        return V_new

    def bellman_solve(self, tol=1e-6, max_iter=1000, howard_iter=20):
        V_old = self.V_0()
        error = np.inf
        iteration = 0

        while error >= tol and iteration < max_iter:

            # expensive step: update policy
            V_new, s_policy = self.bellman_operator(V_old)

            # cheap steps: evaluate fixed policy
            for _ in range(howard_iter):
                V_new = self.policy_evaluation_step(V_new, s_policy)

            error = np.max(np.abs(V_new - V_old))
            V_old = V_new
            iteration += 1

            if iteration % 5 == 0:
                print(f"Iteration {iteration}, error = {error:.3e}")

        print(f"Stopped after {iteration} iterations, error = {error:.3e}")

        self.V = V_new
        self.s_policy = s_policy

        return self

    def map_to_decentralised(self):

        # m(s,q) = s + q - s'(s,q)
        m = self.s_grid[:, None] + self.q_grid[None, :] - self.s_policy
        self.m_policy = m
        # the commodity price p(s,q) = F_m(m(s,q))
        p_s = self.marginal_product(m)
        self.price_s = p_s
        # get MU of consumption at the optimal policy
        c = self.production(m)
        u_c = self.marginal_utility(c)
        # p_s(s,q) u_c(c)= beta * E[ u_c(c') p_s(s',q')|s,q]+delta(s,q)
        # A = beta * E[A'|s,q] + delta(s,q)
        A = u_c * p_s
        A_interp = [
            interp1d(
                self.s_grid,
                A[:, i_q],
                kind="linear",
                fill_value="extrapolate",
            )
            for i_q in range(self.n_q)
        ]
        A_next = np.zeros((self.n_s, self.n_q, self.n_q))
        for i_q_next in range(self.n_q):
            A_next[:, :, i_q_next] = A_interp[i_q_next](self.s_policy)
        # expectation over q' conditional on current q
        EA = np.einsum("ijk,jk->ij", A_next, self.P_q)
        delta = p_s - self.beta * EA / u_c
        self.convenience_yield = np.maximum(delta, 0.0)
        self.expected_Mp_next = p_s - delta

        return self

    def eval_next_by_q(self, X):
        """
        Given X with shape (n_s, n_q), return X_next with shape (n_s, n_q, n_q),
        where X_next[i_s, i_q, j_q] = X(s_policy[i_s, i_q], q'_j).
        """

        X_next = np.zeros((self.n_s, self.n_q, self.n_q))

        for j_q in range(self.n_q):
            interp = interp1d(
                self.s_grid,
                X[:, j_q],
                kind="linear",
                fill_value="extrapolate",
            )

            X_next[:, :, j_q] = interp(self.s_policy)

        return X_next

    def compute_futures_curves(self, T=12):
        n_s, n_q = self.n_s, self.n_q

        c = self.production(self.m_policy)
        uc = self.marginal_utility(c)

        F = np.zeros((n_s, n_q, T + 1))
        Ep = np.zeros((n_s, n_q, T + 1))
        wedge = np.zeros((n_s, n_q, T + 1))

        B_prev = self.price_s.copy()
        D_prev = np.ones_like(self.price_s)
        Ep_prev = self.price_s.copy()

        F[:, :, 0] = self.price_s
        Ep[:, :, 0] = self.price_s

        for h in range(1, T + 1):

            uc_next = self.eval_next_by_q(uc)
            B_next = self.eval_next_by_q(B_prev)
            D_next = self.eval_next_by_q(D_prev)
            Ep_next = self.eval_next_by_q(Ep_prev)

            # M(s,q,q') = beta * uc(s',q') / uc(s,q)
            M_next = self.beta * uc_next / uc[:, :, None]

            # conditional probabilities P(q'|q), shape needs to broadcast over s
            P = self.P_q[None, :, :]

            B_new = np.sum(P * M_next * B_next, axis=2)
            D_new = np.sum(P * M_next * D_next, axis=2)
            Ep_new = np.sum(P * Ep_next, axis=2)

            F[:, :, h] = B_new / D_new
            Ep[:, :, h] = Ep_new
            wedge[:, :, h] = F[:, :, h] - Ep[:, :, h]

            B_prev = B_new
            D_prev = D_new
            Ep_prev = Ep_new

        self.futures_curves = F
        self.expected_spots = Ep
        self.futures_wedge = wedge

        return self


# run the model and plot results -------------------------------------------------------

# %%
# initialise model
model = CommodityModel(
    crra=2.0,
    beta=0.95,
    alpha=0.5,
    n_storage_states=200,
    inflow_mean=1.0,
    inflow_rho=0.9,
    inflow_sigma=0.2,
    n_inflow_states=100,
    seed=42,
)
# solve
model.bellman_solve().map_to_decentralised()

# plots ----------------------------------


# %%

# pick some q states to plot
indices = [0, 6, 9]

# policy function s'(s, q = j) ---------------------
plt.figure(figsize=(8, 5))
for i_q in indices:
    plt.plot(
        model.s_grid, model.s_policy[:, i_q], label=rf"$q = {model.q_grid[i_q]:.2f}$"
    )
plt.xlim(0, 0.2)
plt.ylim(0, 0.2)
plt.xlabel(r"$s$")
plt.ylabel(r"$s'(s,q)$")
plt.title(r"Policy function $s'(s,q)$ for fixed $q$")
plt.legend()
plt.tight_layout()
plt.savefig("s_policy.png", dpi=300)
plt.close()

# input to production m(s,q = j) = s + j - s'(s, q = j) ---------
plt.figure(figsize=(8, 5))
for i_q in indices:
    m = model.s_grid + model.q_grid[i_q] - model.s_policy[:, i_q]
    plt.plot(model.s_grid, m, label=rf"$q = {model.q_grid[i_q]:.2f}$")
plt.xlim(0, 0.2)
plt.ylim(0, 0.2)
plt.xlabel(r"$s$")
plt.ylabel(r"$m(s,q)$")
plt.title(r"Commodity employed for production $m(s,q)$ for fixed $q$")
plt.legend()
plt.tight_layout()
plt.savefig("m_policy.png", dpi=300)
plt.close()

# where the constraint binds --------
binding = model.s_policy < 1e-5
plt.figure(figsize=(8, 5))
plt.imshow(
    binding.T,
    origin="lower",
    aspect="auto",
    extent=[model.s_grid[0], model.s_grid[-1], model.q_grid[0], model.q_grid[-1]],
)
plt.xlabel("s")
plt.ylabel("q")
plt.title("Region where storage constraint binds: s'(s,q)=0")
plt.savefig("binding_heatmap.png", dpi=300)
plt.close()

# storage heatmap ----------
plt.figure(figsize=(8, 5))
plt.imshow(
    model.s_policy.T,
    origin="lower",
    aspect="auto",
    extent=[model.s_grid[0], model.s_grid[-1], model.q_grid[0], model.q_grid[-1]],
)
plt.xlabel("s")
plt.ylabel("q")
plt.title("Optimal storage policy s'(s,q)")
plt.colorbar(label="s'")
plt.savefig("storage_heatmap.png", dpi=300)
plt.close()

# commodity spot heatmap ----------
plt.figure(figsize=(8, 5))
plt.imshow(
    model.price_s.T,
    origin="lower",
    aspect="auto",
    extent=[model.s_grid[0], model.s_grid[-1], model.q_grid[0], model.q_grid[-1]],
)
plt.xlabel("s")
plt.ylabel("q")
plt.title(r"Commodity price function $p_s(s,q)$")
plt.colorbar(label=r"$p_s$")
plt.savefig("commodity_spot_heatmap.png", dpi=300)
plt.close()

# convenience yield spot heatmap ----------
plt.figure(figsize=(8, 5))
plt.imshow(
    model.convenience_yield.T,
    origin="lower",
    aspect="auto",
    extent=[model.s_grid[0], model.s_grid[-1], model.q_grid[0], model.q_grid[-1]],
)
plt.xlabel("s")
plt.ylabel("q")
plt.title(r"Convenience yield $\delta(s,q)$")
plt.colorbar(label=r"$\delta$")
plt.savefig("convenience_yield_heatmap.png", dpi=300)
plt.close()


# %%

# building futures curves
T = 3
model.compute_futures_curves(T=T)
# plot futures curves for some (s,q) pairs
plt.figure(figsize=(8, 5))
for i_q in [0]:
    for i_s in [0, 9, 19]:
        F = model.futures_curves[i_s, i_q, :]
        plt.plot(
            range(T + 1),
            F,
            label=rf"$s={model.s_grid[i_s]:.3f}, q={model.q_grid[i_q]:.2f}$",
        )
plt.xlabel("Maturity (months)")
plt.ylabel("Futures price")
plt.title("Futures curves for different (s,q) pairs")
plt.legend()
plt.tight_layout()
plt.savefig("futures_curves.png", dpi=300)
plt.close()

# plot futures curves for some (s,q) pairs
for i in range(1, T + 1):
    plt.figure(figsize=(8, 5))
    plt.imshow(
        model.futures_wedge[:, :, i].T,
        origin="lower",
        aspect="auto",
        extent=[model.s_grid[0], model.s_grid[-1], model.q_grid[0], model.q_grid[-1]],
    )
    plt.xlabel("s")
    plt.ylabel("q")
    plt.title(
        "Futures wedge, futures price minus expected spot, for maturity " + str(i)
    )
    plt.colorbar(label=r"$F - E[p|s,q]$")
    plt.tight_layout()
    plt.savefig(f"futures_wedge_{i}.png", dpi=300)
    plt.close()

# %%
