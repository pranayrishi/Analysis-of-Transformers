"""
Single-DOF structural oscillator data generation (Section 3.1, Eq. 12).

  m ẍ + c ẋ + k x = 0
  x(0) = x0,  ẋ(0) = v0

Two stiffness cases are studied:
  Case 1:  k = 2000  →  f_n ≈ 7.12 Hz   (both AR coefficients < 0)
  Case 2:  k =  500  →  f_n ≈ 3.56 Hz   (mixed-sign AR coefficients)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def generate_sdof_data(
    m: float = 1.0,
    c: float = 0.5,
    k: float = 2000.0,
    x0: float = 0.01,
    v0: float = 0.0,
    t_end: float = 10.0,
    fs: float = 25.0,
):
    """Generate free-vibration displacement (and velocity) at *fs* Hz.

    Returns
    -------
    t : 1-D array     Time vector (s).
    x : 1-D array     Displacement (m).
    v : 1-D array     Velocity (m/s) — not observed, kept for reference.
    """
    dt = 1.0 / fs
    t_eval = np.arange(0, t_end, dt)

    def rhs(t, y):
        return [y[1], (-c * y[1] - k * y[0]) / m]

    sol = solve_ivp(
        rhs, [0, t_end], [x0, v0],
        t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-12,
    )
    return sol.t, sol.y[0], sol.y[1]


def compute_sdof_ar2_coefficients(
    m: float, c: float, k: float, dt: float
):
    """Exact discrete-time AR(2) coefficients via matrix exponential.

    For the state-space  ẏ = A y  with  A = [[0, 1], [-k/m, -c/m]]:
      Φ = expm(A Δt)
      c_1 = tr(Φ),   c_2 = -det(Φ)

    such that  x_{n+1} = c_1 x_n + c_2 x_{n-1}   (Eq. 11 regime).
    """
    A_cont = np.array([[0.0, 1.0], [-k / m, -c / m]])
    Phi = expm(A_cont * dt)
    c1 = np.trace(Phi)
    c2 = -np.linalg.det(Phi)
    return c1, c2
