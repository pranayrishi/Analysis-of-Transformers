"""
Van der Pol oscillator data generation (Section 4.1, Eq. 15).

  ẍ − μ(1 − x²) ẋ + x = 0        μ = 0.5

Training: 1500 trajectories, ICs ~ U(-3,3)², t ∈ [0, 6.5], Δt = 0.1.
Test:     single long trajectory from (2, 0), t ∈ [0, 65].
"""

import numpy as np
from scipy.integrate import solve_ivp


def _vdp_rhs(t, y, mu):
    x, v = y
    return [v, mu * (1 - x ** 2) * v - x]


def generate_vanderpol_data(
    mu: float = 0.5,
    n_trajectories: int = 1500,
    t_end: float = 6.5,
    dt: float = 0.1,
    x_range: tuple = (-3, 3),
    seed: int = 42,
):
    """Generate training / validation trajectories.

    Returns
    -------
    trajectories : list[ndarray (T, 2)]   Each row is (x, ẋ).
    t_eval       : 1-D array
    """
    rng = np.random.default_rng(seed)
    t_eval = np.arange(0, t_end, dt)

    trajectories = []
    for _ in range(n_trajectories):
        x0 = rng.uniform(*x_range)
        v0 = rng.uniform(*x_range)
        sol = solve_ivp(
            _vdp_rhs, [0, t_end], [x0, v0],
            args=(mu,), t_eval=t_eval, method="BDF",
            rtol=1e-6, atol=1e-9,
        )
        if sol.success:
            trajectories.append(np.column_stack([sol.y[0], sol.y[1]]))
    return trajectories, t_eval


def generate_vanderpol_test_trajectory(
    mu: float = 0.5, t_end: float = 65.0, dt: float = 0.1
):
    """Long test trajectory starting near the limit cycle at (2, 0).

    Returns
    -------
    traj : ndarray (T, 2)
    t_eval : 1-D array
    """
    t_eval = np.arange(0, t_end, dt)
    sol = solve_ivp(
        _vdp_rhs, [0, t_end], [2.0, 0.0],
        args=(mu,), t_eval=t_eval, method="BDF",
        rtol=1e-6, atol=1e-9,
    )
    return np.column_stack([sol.y[0], sol.y[1]]), t_eval
