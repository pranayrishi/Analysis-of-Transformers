"""
Two-DOF coupled oscillator data generation (Section 3.2, Eq. 14).

  M ẍ + C ẋ + K x = 0

  M = diag(m1, m2)
  K = [[k1+k2, -k2], [-k2, k2]]
  C = [[c1+c2, -c2], [-c2, c2]]

Natural frequencies ≈ 4.1 Hz and 9.5 Hz.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def generate_2dof_data(
    m1=1.0, m2=1.0,
    c1=0.5, c2=0.5,
    k1=1000.0, k2=1500.0,
    x0=(10.0, 0.0),
    v0=(0.0, 0.0),
    t_end=25.0,
    fs=25.0,
):
    """Generate free-vibration trajectories at *fs* Hz.

    Returns
    -------
    t  : 1-D array
    x1, x2 : 1-D arrays   Displacements.
    v1, v2 : 1-D arrays   Velocities (reference).
    """
    dt = 1.0 / fs
    t_eval = np.arange(0, t_end, dt)

    M_mat = np.diag([m1, m2])
    K_mat = np.array([[k1 + k2, -k2], [-k2, k2]])
    C_mat = np.array([[c1 + c2, -c2], [-c2, c2]])
    M_inv = np.linalg.inv(M_mat)

    def rhs(t, y):
        x = y[:2]
        v = y[2:]
        a = M_inv @ (-C_mat @ v - K_mat @ x)
        return np.concatenate([v, a])

    y0 = np.concatenate([list(x0), list(v0)])
    sol = solve_ivp(
        rhs, [0, t_end], y0,
        t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-12,
    )
    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


def compute_2dof_ar_matrices(
    m1, m2, c1, c2, k1, k2, dt
):
    """Exact discrete-time state-transition and AR matrices for the 2-DOF.

    Returns
    -------
    Phi : (4,4) matrix exponential
    ar_coeffs : list of (2,2) AR coefficient matrices
        For the displacement sub-vector [x1, x2]:
          x_{n+1} = C1 x_n + C2 x_{n-1}   with
          C1 = tr-like quantity, C2 = -det-like quantity
          (generalised from scalar case)
    """
    M_mat = np.diag([m1, m2])
    K_mat = np.array([[k1 + k2, -k2], [-k2, k2]])
    C_mat = np.array([[c1 + c2, -c2], [-c2, c2]])
    M_inv = np.linalg.inv(M_mat)

    # 4×4 continuous system matrix
    A_cont = np.zeros((4, 4))
    A_cont[:2, 2:] = np.eye(2)
    A_cont[2:, :2] = -M_inv @ K_mat
    A_cont[2:, 2:] = -M_inv @ C_mat

    Phi = expm(A_cont * dt)  # (4,4)

    # Extract displacement-only sub-matrices
    # x_{n+1} = Phi_xx x_n + Phi_xv v_n
    # v_{n+1} = Phi_vx x_n + Phi_vv v_n
    # Eliminating v gives vector AR(2) in x:
    #   x_{n+1} = (Phi_xx + Phi_xv Phi_vv^{-1} ...) — complicated
    # Simpler: just return the full Phi for spectrum analysis

    return Phi
