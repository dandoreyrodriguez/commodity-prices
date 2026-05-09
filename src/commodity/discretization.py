import numpy as np
from scipy.stats import norm


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