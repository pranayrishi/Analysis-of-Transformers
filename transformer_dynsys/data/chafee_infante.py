"""
Chafee-Infante reaction-diffusion PDE (Section 4.2, Eq. 16-18).

  u_t = u - u^3 + nu u_{xx},   u(0,t) = u(pi,t) = 0,   nu = 0.16

Galerkin projection onto 3 sine modes:
  u(x,t) ~ sum_{k=1}^{3} phi_k(t) sin(kx)

The cubic Galerkin coupling tensor G_{jklm} is PRECOMPUTED once at
import time (only 81 integrals) so that the ODE RHS is fast.
"""

import numpy as np
from scipy.integrate import solve_ivp, quad


# ═══════════════════════════════════════════════════════════════════
#  Precomputed cubic coupling tensor
# ═══════════════════════════════════════════════════════════════════

def _precompute_cubic_coefficients(n_modes: int = 3) -> np.ndarray:
    """Compute  G_{jklm} = (2/pi) int_0^pi sin(jx)sin(kx)sin(lx)sin(mx) dx.

    Only 3^4 = 81 integrals — takes milliseconds.
    """
    G = np.zeros((n_modes, n_modes, n_modes, n_modes))
    for j in range(n_modes):
        for k in range(n_modes):
            for l in range(n_modes):
                for m in range(n_modes):
                    def integrand(x, _j=j, _k=k, _l=l, _m=m):
                        return (np.sin((_j + 1) * x) * np.sin((_k + 1) * x) *
                                np.sin((_l + 1) * x) * np.sin((_m + 1) * x))
                    val, _ = quad(integrand, 0, np.pi)
                    G[j, k, l, m] = val * 2.0 / np.pi
    return G


_G_TENSOR = _precompute_cubic_coefficients(3)


# ═══════════════════════════════════════════════════════════════════
#  ODE right-hand side (fast — no quadrature at runtime)
# ═══════════════════════════════════════════════════════════════════

def chafee_infante_rhs(t, phi, nu=0.16):
    """3-mode Galerkin ODE:
    dphi_m/dt = (1 - nu m^2) phi_m - sum_{j,k,l} G_{jklm} phi_j phi_k phi_l
    """
    p = np.asarray(phi)
    n_modes = len(p)
    dpdt = np.zeros(n_modes)
    for m in range(n_modes):
        linear = (1.0 - nu * (m + 1) ** 2) * p[m]
        cubic = 0.0
        for j in range(n_modes):
            for k in range(n_modes):
                for l in range(n_modes):
                    cubic += _G_TENSOR[j, k, l, m] * p[j] * p[k] * p[l]
        dpdt[m] = linear - cubic
    return dpdt


# ═══════════════════════════════════════════════════════════════════
#  Trajectory generation
# ═══════════════════════════════════════════════════════════════════

def generate_chafee_infante_trajectories(
    nu: float = 0.16,
    n_trajectories: int = 500,
    t_end: float = 4.0,
    n_time: int = 10,
    phi_range: float = 1.5,
    seed: int = 42,
    discard_transient: bool = True,
    transient_time: float = 10.0,
):
    """Generate short Galerkin-mode trajectories.

    Parameters
    ----------
    n_trajectories : int
    t_end : float        Integration horizon per trajectory.
    n_time : int         Number of saved time-points per trajectory.
    phi_range : float    ICs for phi1, phi2 ~ U(-phi_range, phi_range).
    discard_transient : bool
        Pre-integrate for *transient_time* to reach the inertial manifold.

    Returns
    -------
    trajectories : list[ndarray (n_time, 3)]
    t_eval : 1-D array
    """
    rng = np.random.default_rng(seed)
    t_eval = np.linspace(0, t_end, n_time)

    trajectories = []
    for _ in range(n_trajectories):
        phi1 = rng.uniform(-phi_range, phi_range)
        phi2 = rng.uniform(-0.6, 0.6)
        phi3 = rng.uniform(-0.3, 0.3)
        y0 = [phi1, phi2, phi3]

        # Discard transient so dynamics settle onto inertial manifold
        if discard_transient:
            sol_t = solve_ivp(
                lambda t, y: chafee_infante_rhs(t, y, nu),
                [0, transient_time], y0,
                method="RK45", rtol=1e-8, atol=1e-10,
            )
            if not sol_t.success:
                continue
            y0 = sol_t.y[:, -1].tolist()

        sol = solve_ivp(
            lambda t, y: chafee_infante_rhs(t, y, nu),
            [0, t_end], y0,
            t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10,
        )
        if sol.success:
            trajectories.append(sol.y.T)  # (n_time, 3)

    return trajectories, t_eval


# ═══════════════════════════════════════════════════════════════════
#  Physical-space reconstruction & observation extraction
# ═══════════════════════════════════════════════════════════════════

def reconstruct_physical_space(phi: np.ndarray, n_spatial: int = 256):
    """u(x, t) = sum phi_k(t) sin(k x) on uniform grid [0, pi].

    Returns x_grid (n_spatial,) and u (T, n_spatial).
    """
    x_grid = np.linspace(0, np.pi, n_spatial, endpoint=False)
    T, n_modes = phi.shape
    u = np.zeros((T, n_spatial))
    for k in range(n_modes):
        u += phi[:, k : k + 1] * np.sin((k + 1) * x_grid[None, :])
    return x_grid, u


def extract_observation(phi: np.ndarray, grid_index: int = 10,
                        n_spatial: int = 256) -> np.ndarray:
    """Scalar observation u(x_{grid_index}, t) from Fourier modes."""
    x_pt = grid_index * np.pi / n_spatial
    return sum(phi[:, k] * np.sin((k + 1) * x_pt) for k in range(phi.shape[1]))
