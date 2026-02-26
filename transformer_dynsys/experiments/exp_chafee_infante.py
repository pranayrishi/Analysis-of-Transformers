"""
Section 4.2 — Chafee-Infante reaction-diffusion experiments.

Reproduces **Figures 6-8**:
  Fig 6a: 3D scatter of Fourier modes (ϕ₁, ϕ₂, ϕ₃)
  Fig 6b: Physical-space reconstruction u(x,t)
  Fig 6c: |MSE| comparison across 5 model types × 10 seeds
  Fig 7a: Latent space variables for representative 3D model
  Fig 7b-c: Attention matrices across seeds with/without P.E.
  Fig 8a-b: 2D projections of z+u^{emb} coloured by Fourier modes
  Fig 8c-d: Attention matrices for 3D models
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import DataLoader

from utils.helpers import set_seed, get_device
from data.chafee_infante import (
    generate_chafee_infante_trajectories,
    reconstruct_physical_space,
    extract_observation,
)
from models.transformer import SingleLayerTransformer
from models.mlp import MLPBaseline
from training.dataset import WindowedTimeSeriesDataset, MLPDataset
from training.trainer import Trainer

DEVICE = get_device()
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SEQ_LEN = 5
EPOCHS = 1000
LR = 1e-3
BATCH_SIZE = 64
MLP_HIDDEN = 64
NUM_SEEDS = 10
N_SPATIAL = 256
OBS_IDX = 10  # 10th grid point on [0, π]


# ── data preparation ────────────────────────────────────────────────
def prepare_data():
    trajs_modes, t_eval = generate_chafee_infante_trajectories(
        nu=0.16, n_trajectories=500, t_end=4.0, n_time=10, seed=42
    )
    # Extract scalar observation from each trajectory
    obs_trajs = []
    for phi in trajs_modes:
        u_obs = extract_observation(phi, grid_index=OBS_IDX, n_spatial=N_SPATIAL)
        obs_trajs.append(u_obs[:, None])  # (T, 1)

    n = len(obs_trajs)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train = obs_trajs[:n_train]
    val = obs_trajs[n_train:n_train + n_val]
    test = obs_trajs[n_train + n_val:]

    return trajs_modes, train, val, test, t_eval


# ── training wrappers ───────────────────────────────────────────────
def train_transformer_ci(train, val, d_inner, use_pe, seed):
    set_seed(seed)
    train_ds = WindowedTimeSeriesDataset(train, SEQ_LEN)
    val_ds = WindowedTimeSeriesDataset(val, SEQ_LEN)
    model = SingleLayerTransformer(
        d_obs=1, d_inner=d_inner, d_output=1, seq_len=SEQ_LEN,
        use_mlp=True, mlp_hidden=MLP_HIDDEN,
        use_positional_encoding=use_pe, use_residual=True,
    )
    trainer = Trainer(model, train_ds, val_ds,
                      lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS,
                      device=DEVICE, verbose=False)
    return trainer.train()


def train_mlp_ci(train, val, seed):
    set_seed(seed)
    train_ds = MLPDataset(train)
    val_ds = MLPDataset(val)
    model = MLPBaseline(1, 1, hidden_size=MLP_HIDDEN)
    trainer = Trainer(model, train_ds, val_ds,
                      lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS,
                      device=DEVICE, verbose=False)
    return trainer.train()


def compute_test_mse(model, test_trajs, is_transformer=True):
    model.eval()
    if is_transformer:
        ds = WindowedTimeSeriesDataset(test_trajs, SEQ_LEN)
    else:
        ds = MLPDataset(test_trajs)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    total, n = 0.0, 0
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            if is_transformer:
                y_pred, _ = model(X_b)
            else:
                y_pred = model(X_b)
            total += ((y_pred - y_b) ** 2).sum().item()
            n += y_b.numel()
    return total / n


# ── Figure 6a: 3D Fourier-mode DENSE scatter ──────────────────────
def plot_figure6a(trajs_modes):
    """Dense scatter cloud of the inertial manifold in (ϕ₁, ϕ₂, ϕ₃) space.

    Uses a SEPARATE high-resolution dataset for visualisation only
    (many trajectories × many time points) so the manifold surface
    is clearly visible as a dense point cloud — NOT connected trajectory lines.
    """
    from data.chafee_infante import generate_chafee_infante_trajectories

    # Generate dense data for scatter visualisation.
    # Use SHORT transient (1.0s) so φ₃ has decayed but the trajectory
    # is still approaching the equilibria along the inertial manifold.
    # This captures the 2D manifold surface, not just the fixed points.
    dense_trajs, _ = generate_chafee_infante_trajectories(
        nu=0.16, n_trajectories=3000, t_end=6.0, n_time=80,
        seed=99, discard_transient=True, transient_time=1.0,
    )
    # Concatenate ALL points from ALL trajectories into one array
    all_pts = np.vstack(dense_trajs)  # (N_total, 3)
    print(f"  Figure 6a: {all_pts.shape[0]} scatter points")

    phi1, phi2, phi3 = all_pts[:, 0], all_pts[:, 1], all_pts[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # SCATTER plot — visible dots, semi-transparent
    ax.scatter(phi1, phi2, phi3,
               c=phi1, cmap="coolwarm",
               s=2, alpha=0.4, rasterized=True)

    # Projection onto (φ₁, φ₂) plane at bottom
    phi3_floor = phi3.min() - 0.05
    ax.scatter(phi1, phi2,
               np.full_like(phi3, phi3_floor),
               c="lightgray", s=0.5, alpha=0.15, rasterized=True)

    # Highlight ONE reference trajectory as a connected blue line
    if len(dense_trajs) > 0:
        ref = dense_trajs[0]
        ax.plot(ref[:, 0], ref[:, 1], ref[:, 2],
                "o-", color="dodgerblue", linewidth=1.5, markersize=4,
                alpha=0.9, label="Reference trajectory", zorder=10)

    ax.set_xlabel("$\\phi_1$")
    ax.set_ylabel("$\\phi_2$")
    ax.set_zlabel("$\\phi_3$")
    ax.set_title("Figure 6a — Chafee-Infante: Inertial Manifold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "figure6a_ci_modes.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Figure 6b: physical-space 3D surface ─────────────────────────────
def plot_figure6b(trajs_modes):
    """3D surface plot of u(x, t) for a representative trajectory."""
    # Use a trajectory with longer time resolution for a smooth surface
    from data.chafee_infante import generate_chafee_infante_trajectories

    hi_res_trajs, t_eval_hr = generate_chafee_infante_trajectories(
        nu=0.16, n_trajectories=5, t_end=4.0, n_time=80,
        seed=42, discard_transient=True, transient_time=10.0,
    )
    ref = hi_res_trajs[0]  # (80, 3)
    x_grid, u = reconstruct_physical_space(ref, N_SPATIAL)

    T_grid, X_grid = np.meshgrid(t_eval_hr, x_grid)
    # u is (n_time, n_spatial), surface needs (n_spatial, n_time)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(T_grid, X_grid, u.T,
                    cmap="RdBu_r", edgecolor="none", alpha=0.9,
                    rstride=2, cstride=2)
    # Mark observation point as a line on the surface
    x_obs = OBS_IDX * np.pi / N_SPATIAL
    obs_idx = np.argmin(np.abs(x_grid - x_obs))
    ax.plot(t_eval_hr, [x_grid[obs_idx]] * len(t_eval_hr), u[:, obs_idx],
            "r-", linewidth=2, label=f"Obs (x≈{x_obs:.3f})")

    ax.set_xlabel("Time")
    ax.set_ylabel("x")
    ax.set_zlabel("u(x, t)")
    ax.set_title("Figure 6b — Physical Space u(x, t)")
    ax.legend(fontsize=8)
    path = os.path.join(FIG_DIR, "figure6b_ci_physical.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Figure 6c: MSE comparison ──────────────────────────────────────
def run_mse_comparison(train, val, test):
    configs = {
        "MLP Only": dict(is_tf=False),
        "MLP+Attn 2D\n(No P.E.)": dict(d_inner=2, pe=False),
        "MLP+Attn 3D\n(No P.E.)": dict(d_inner=3, pe=False),
        "MLP+Attn 2D\n(P.E.)": dict(d_inner=2, pe=True),
        "MLP+Attn 3D\n(P.E.)": dict(d_inner=3, pe=True),
    }
    results = {k: [] for k in configs}

    for seed in range(NUM_SEEDS):
        print(f"  Seed {seed}")
        for name, cfg in configs.items():
            if cfg.get("is_tf") is False:
                m = train_mlp_ci(train, val, seed)
                mse = compute_test_mse(m, test, False)
            else:
                m = train_transformer_ci(train, val, cfg["d_inner"], cfg["pe"], seed)
                mse = compute_test_mse(m, test, True)
            results[name].append(mse)

    # plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, mses) in enumerate(results.items()):
        mses = np.array(mses)
        ax.bar(i, mses.mean(), width=0.5, alpha=0.5, color=f"C{i}")
        ax.scatter([i] * len(mses), mses, s=20, c=f"C{i}", zorder=3)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(results.keys(), rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("|MSE|")
    ax.set_yscale("log")
    ax.set_title("Figure 6c — Chafee-Infante |MSE|")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "figure6c_ci_mse.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")

    return results


# ── Figure 7-8: latent analysis ─────────────────────────────────────
def plot_latent_analysis(train, val, test, trajs_modes):
    """Figures 7 and 8: latent space unfolding and attention.

    Uses a SEPARATE high-resolution dataset (500 trajs × 80 time steps)
    so the latent manifold structure is visible with thousands of points.
    """
    # Train representative models on the original (sparse) training data
    model_3d = train_transformer_ci(train, val, d_inner=3, use_pe=True, seed=0)
    model_2d = train_transformer_ci(train, val, d_inner=2, use_pe=True, seed=0)

    # ── Generate DENSE data for latent visualisation ─────────────────
    # Short transient (1.0s) so dynamics are still approaching equilibria
    # along the inertial manifold — captures the 2D surface, not just
    # the fixed points.
    dense_trajs, _ = generate_chafee_infante_trajectories(
        nu=0.16, n_trajectories=500, t_end=4.0, n_time=80,
        seed=999, discard_transient=True, transient_time=1.0,
    )
    # Convert Fourier modes to scalar observations
    dense_obs = []
    for phi in dense_trajs:
        u_obs = extract_observation(phi, grid_index=OBS_IDX, n_spatial=N_SPATIAL)
        dense_obs.append(u_obs[:, None])  # (80, 1)

    # Build aligned Fourier-mode array for colouring.
    # Each trajectory of length 80 produces 75 windows (80 - SEQ_LEN).
    # Window t uses obs[t:t+SEQ_LEN] as input; the corresponding
    # Fourier mode for colouring is phi[t + SEQ_LEN] (the target time).
    all_modes = []
    for phi in dense_trajs:
        for t in range(len(phi) - SEQ_LEN):
            all_modes.append(phi[t + SEQ_LEN])
    all_modes = np.array(all_modes)  # (N_windows, 3)

    print(f"  Dense latent data: {len(dense_obs)} trajs × {dense_obs[0].shape[0]} steps"
          f" → {all_modes.shape[0]} windows")

    # Collect internals in batches
    def get_internals(model, obs_trajs):
        ds = WindowedTimeSeriesDataset(obs_trajs, SEQ_LEN)
        loader = DataLoader(ds, batch_size=2048, shuffle=False)
        all_ints = {}
        model.eval()
        with torch.no_grad():
            for X_b, _ in loader:
                X_b = X_b.to(DEVICE)
                _, ints = model(X_b)
                for k, v in ints.items():
                    arr = v.cpu().numpy()
                    if k not in all_ints:
                        all_ints[k] = [arr]
                    else:
                        all_ints[k].append(arr)
        return {k: np.concatenate(v, axis=0) for k, v in all_ints.items()}

    ints_3d = get_internals(model_3d, dense_obs)
    ints_2d = get_internals(model_2d, dense_obs)

    # ── Figure 7a: latent variables scatter ──────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Figure 7a — 3D Latent Variables", fontsize=13)

    z_last = ints_3d["Z"][:, -1, :]     # (N, 3) — pure attention (before residual)
    x_emb = ints_3d["X_emb"][:, -1, :]  # (N, 3)
    z_plus = ints_3d["Z_residual"][:, -1, :]  # (N, 3) — attention + residual

    variables = [
        (x_emb, "$x^{emb}$"),
        (ints_3d["Q"][:, -1, :], "$Q$"),
        (ints_3d["K"][:, -1, :], "$K$"),
        (z_plus, "$Z + x^{emb}$"),
    ]
    for ax, (data, lbl) in zip(axes, variables):
        if data.shape[1] >= 2:
            sc = ax.scatter(data[:, 0], data[:, 1],
                            c=all_modes[:len(data), 0], cmap="coolwarm",
                            s=1, alpha=0.4, rasterized=True)
            plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(lbl)
        ax.set_xlabel(f"{lbl}$_1$")
        ax.set_ylabel(f"{lbl}$_2$")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "figure7a_ci_latent.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ── Figure 8: 2D projections coloured by Fourier modes ──────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Figure 8 — 3D Latent Projections (coloured by Fourier modes)", fontsize=13)

    n_pts = min(len(z_plus), len(all_modes))
    panels = [
        (0, 1, 0, "$\\phi_1$"),  # (Z1, Z2) coloured by φ₁
        (0, 1, 1, "$\\phi_2$"),  # (Z1, Z2) coloured by φ₂
        (1, 2, 2, "$\\phi_3$"),  # (Z2, Z3) coloured by φ₃
    ]
    for ax, (di, dj, ck, clbl) in zip(axes, panels):
        sc = ax.scatter(z_plus[:n_pts, di], z_plus[:n_pts, dj],
                        c=all_modes[:n_pts, ck], cmap="coolwarm",
                        s=1, alpha=0.4, rasterized=True)
        ax.set_xlabel(f"$z_{di+1}+x^{{emb}}_{di+1}$")
        ax.set_ylabel(f"$z_{dj+1}+x^{{emb}}_{dj+1}$")
        ax.set_title(f"Coloured by {clbl}")
        plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "figure8_ci_projections.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ── Figure 7b-c: Attention heatmaps ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Attention Matrices", fontsize=13)

    for ax_i, (model, ints, label) in enumerate([
        (model_3d, ints_3d, "3D + P.E."),
        (model_2d, ints_2d, "2D + P.E."),
    ]):
        A = ints["A"]  # (N, 5, 5)
        # Sub-sample evenly to show ~40 representative attention patterns
        n_show = min(40, A.shape[0])
        step = max(1, A.shape[0] // n_show)
        last_rows = A[::step, -1, :][:n_show]  # (n_show, 5)
        im = axes[ax_i].imshow(last_rows, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        axes[ax_i].set_xlabel("Token position")
        axes[ax_i].set_ylabel("Sample")
        axes[ax_i].set_title(f"Attention: {label}")
        plt.colorbar(im, ax=axes[ax_i])

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "figure7bc_ci_attention.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ── main ────────────────────────────────────────────────────────────
def main():
    print("Generating Chafee-Infante data...")
    trajs_modes, train, val, test, t_eval = prepare_data()

    plot_figure6a(trajs_modes)
    plot_figure6b(trajs_modes)

    print("\n=== MSE comparison (Figure 6c) ===")
    run_mse_comparison(train, val, test)

    print("\n=== Latent analysis (Figures 7-8) ===")
    plot_latent_analysis(train, val, test, trajs_modes)

    print("\nChafee-Infante experiments complete.")


if __name__ == "__main__":
    main()
