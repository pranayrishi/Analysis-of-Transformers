"""
Default hyperparameters for reproducing:
"A Mechanistic Analysis of Transformers for Dynamical Systems"
Duthé et al. (arXiv:2512.21113v1, December 2025)
"""

# ── Training defaults ──────────────────────────────────────────────
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 1000
NUM_SEEDS = 10          # Paper trains 10 models per configuration

# ── SDOF system (Section 3.1) ─────────────────────────────────────
SDOF = dict(
    m=1.0, c=0.5,
    k_case1=2000.0, k_case2=500.0,
    x0=0.01, v0=0.0,
    t_end=10.0, fs=25.0,
    seq_len=2, d_obs=1, d_inner=1,
)

# ── 2DOF system (Section 3.2) ─────────────────────────────────────
TWODOF = dict(
    m1=1.0, m2=1.0, c1=0.5, c2=0.5,
    k1=1000.0, k2=1500.0,
    x0=[10.0, 0.0], v0=[0.0, 0.0],
    t_end=25.0, fs=25.0,
)

# ── Van der Pol (Section 4.1) ─────────────────────────────────────
VDP = dict(
    mu=0.5,
    n_trajectories=1500,
    t_end_train=6.5, t_end_test=65.0,
    dt=0.1,
    x_range=(-3, 3),
    seq_len=5, d_obs_full=2, d_obs_partial=1,
    train_ratio=0.8, val_ratio=0.1,
)

# ── Chafee-Infante (Section 4.2) ─────────────────────────────────
CI = dict(
    nu=0.16,
    n_spatial=256,
    obs_grid_index=10,       # 10th grid-point on [0, π]
    t_end=4.0, n_time=10,
    seq_len=5,
    train_ratio=0.70, val_ratio=0.15,
)

# ── Navier-Stokes (Section 4.3) ──────────────────────────────────
NS = dict(
    obs_x=35, obs_y=45,
    d_inner=3,
    seq_len=5,
    Re_range=(100, 750),
)
