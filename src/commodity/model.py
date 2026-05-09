import numpy as np

from commodity.discretization import Tauchen
from commodity.kernels import (
    _compute_ee_values,
    _egm_step,
    _egm_interpolate,
    _compute_p_uc_m,
    _convenience_yield_kernel,
    _futures_curve_at_state,
    _futures_surface_fixed_z_at_horizon,
    _futures_surface_fixed_q_at_horizon,
    _futures_surface_fixed_s_at_horizon,
)
from commodity.utils import nearest_index


def _select_object(F, Ep, W, object_name):
    if object_name == "F":
        return F
    if object_name == "Ep":
        return Ep
    if object_name == "wedge":
        return W

    raise ValueError("object_name must be 'F', 'Ep', or 'wedge'")


class CommodityModel:
    def __init__(
        self,
        crra,
        beta,
        alpha,
        n_storage_states=100,
        storage_max_multiple=1.5,
        storage_curvature=3.0,
        inflow_min=1e-5,
        inflow_scale=0.020,
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
        self.inflow_scale = float(inflow_scale)

        self.q_grid = self.inflow_min + self.inflow_scale * np.ascontiguousarray(
            np.exp(self.inflow_process.grid)
        )

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

        self.z_grid = np.ascontiguousarray(
            self.productivity_mean * np.exp(self.productivity_process.grid)
        )

        self.P_z = np.ascontiguousarray(self.productivity_process.P)
        self.stationary_dist_z = self.productivity_process.stationary_dist
        self.n_z = int(n_productivity_states)

        self.n_s = int(n_storage_states)
        self.storage_max_multiple = float(storage_max_multiple)
        self.storage_curvature = float(storage_curvature)

        s_max = self.storage_max_multiple * float(
            self.q_grid @ self.stationary_dist_q
        )

        self.s_grid = np.ascontiguousarray(
            self.make_storage_grid(
                s_max=s_max,
                n_s=self.n_s,
                curvature=self.storage_curvature,
            )
        )

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

    def solve_egm(
        self,
        tol=1e-6,
        max_iter=500,
        verbose=True,
    ):
        s = self.s_grid[:, None, None]
        q = self.q_grid[None, :, None]

        s_policy0 = np.clip(
            0.5 * (s + q),
            0.0,
            self.s_grid[-1],
        )

        s_policy = np.ascontiguousarray(
            np.broadcast_to(
                s_policy0,
                (self.n_s, self.n_q, self.n_z),
            ).copy()
        )

        error = 1.0
        iteration = 0

        while error > tol and iteration < max_iter:

            EE = _compute_ee_values(
                self.s_grid,
                self.q_grid,
                self.z_grid,
                s_policy,
                self.alpha,
                self.crra,
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

            s_policy_new = _egm_interpolate(
                s_implied,
                self.s_grid,
                self.q_grid,
            )

            error = float(np.max(np.abs(s_policy_new - s_policy)))

            s_policy = np.ascontiguousarray(s_policy_new)
            iteration += 1

            if verbose and iteration % 25 == 0:
                print(f"Iteration {iteration}, policy error = {error:.3e}")

        if verbose:
            print(f"Stopped after {iteration} iterations, error = {error:.3e}")

        self.s_policy = s_policy

        self.m_policy = np.maximum(
            self.s_grid[:, None, None]
            + self.q_grid[None, :, None]
            - self.s_policy,
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

    def futures_curve_at_index(
        self,
        i_s,
        i_q,
        i_z,
        T=12,
    ):
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

    def futures_curve_at_values(
        self,
        s,
        q,
        z,
        T=12,
    ):
        i_s = nearest_index(self.s_grid, s)
        i_q = nearest_index(self.q_grid, q)
        i_z = nearest_index(self.z_grid, z)

        out = self.futures_curve_at_index(
            i_s,
            i_q,
            i_z,
            T=T,
        )

        out.update(
            {
                "i_s": i_s,
                "i_q": i_q,
                "i_z": i_z,
                "s": self.s_grid[i_s],
                "q": self.q_grid[i_q],
                "z": self.z_grid[i_z],
            }
        )

        return out

    def futures_surface_fixed_z(
        self,
        i_z,
        T=12,
        object_name="wedge",
    ):
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

    def futures_surface_fixed_q(
        self,
        i_q,
        T=12,
        object_name="wedge",
    ):
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

    def futures_surface_fixed_s(
        self,
        i_s,
        T=12,
        object_name="wedge",
    ):
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