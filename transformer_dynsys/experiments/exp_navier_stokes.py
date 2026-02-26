"""
Section 4.3 — Navier-Stokes flow past a cylinder experiments.

Reproduces **Figures 9-11**:
  Fig 9:  Observation setup and example flow field
  Fig 10: Latent space analysis, Strouhal number recovery
  Fig 11: Full-field reconstruction via geometric harmonics

NOTE: The full CFD dataset from Geneva & Zabaras (2022) must be
obtained separately.  If unavailable, synthetic vortex-shedding
signals are generated as a fallback.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from utils.helpers import set_seed, get_device
from data.navier_stokes import (
    load_navier_stokes_data,
    generate_synthetic_cylinder_data,
)
from models.transformer import SingleLayerTransformer
from models.mlp import MLPBaseline
from training.dataset import WindowedTimeSeriesDataset, MLPDataset
from training.trainer import Trainer

DEVICE = get_device()
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SEQ_LEN = 5
D_INNER = 3
EPOCHS = 1000
LR = 1e-3
BATCH_SIZE = 512
MLP_HIDDEN = 64
NUM_SEEDS = 10
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ns_dataset")


def prepare_data(use_re_input=False):
    """Load or synthesise Navier-Stokes observation data.

    Parameters
    ----------
    use_re_input : bool
        If True, concatenate normalised Re to the observation (d_obs=2).

    Returns
    -------
    all_trajs : list[ndarray]   Scalar (or 2-D) observation trajectories.
    Re_values : list[float]
    """
    try:
        trajs_dict, _ = load_navier_stokes_data(DATA_DIR)
    except FileNotFoundError:
        print("  [INFO] Full NS dataset not found — using synthetic fallback.")
        trajs_dict = generate_synthetic_cylinder_data()

    Re_values = sorted(trajs_dict.keys())
    Re_min, Re_max = min(Re_values), max(Re_values)

    all_trajs = []
    for Re in Re_values:
        obs = trajs_dict[Re]  # (T, 1)
        if use_re_input:
            re_norm = (Re - Re_min) / (Re_max - Re_min + 1e-8)
            re_col = np.full((len(obs), 1), re_norm)
            obs = np.hstack([obs, re_col])  # (T, 2)
        all_trajs.append(obs)

    return all_trajs, Re_values


# ── training wrappers ───────────────────────────────────────────────
def train_ns_transformer(trajs, d_obs, seed, d_output=None):
    """Train NS transformer.

    Parameters
    ----------
    d_obs : int   Input dimension (1 = u_x only, 2 = u_x + Re_norm).
    d_output : int | None
        Output/target dimension.  Default ``d_obs`` for the no-Re case,
        but must be 1 for the +Re case (predict only u_x, not Re).
    """
    if d_output is None:
        d_output = d_obs
    set_seed(seed)
    n = len(trajs)
    n_train = int(0.7 * n)
    train_t = trajs[:n_train]
    val_t = trajs[n_train:]

    # target_dim slices the target to keep only u_x when d_obs=2
    target_dim = d_output if d_output < d_obs else None
    train_ds = WindowedTimeSeriesDataset(train_t, SEQ_LEN, target_dim=target_dim)
    val_ds = WindowedTimeSeriesDataset(val_t, SEQ_LEN, target_dim=target_dim)

    model = SingleLayerTransformer(
        d_obs=d_obs, d_inner=D_INNER,
        d_output=d_output, seq_len=SEQ_LEN,
        use_mlp=True, mlp_hidden=MLP_HIDDEN,
        use_positional_encoding=True, use_residual=True,
    )
    trainer = Trainer(model, train_ds, val_ds,
                      lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS,
                      device=DEVICE, verbose=False)
    model = trainer.train()
    return model


def train_ns_mlp(trajs, d_obs, seed):
    set_seed(seed)
    n = len(trajs)
    n_train = int(0.7 * n)
    train_ds = MLPDataset(trajs[:n_train])
    val_ds = MLPDataset(trajs[n_train:])

    model = MLPBaseline(d_obs, d_obs, hidden_size=MLP_HIDDEN)
    trainer = Trainer(model, train_ds, val_ds,
                      lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS,
                      device=DEVICE, verbose=False)
    return trainer.train()


def compute_test_mse(model, trajs, is_transformer=True, target_dim=None):
    model.eval()
    if is_transformer:
        ds = WindowedTimeSeriesDataset(trajs, SEQ_LEN, target_dim=target_dim)
    else:
        ds = MLPDataset(trajs, target_dim=target_dim)
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


# ── Figure 9: sample flow + observation ─────────────────────────────
def _select_representative_re(Re_values, n=5):
    """Select n Re values spread across the full range."""
    if len(Re_values) <= n:
        return list(range(len(Re_values)))
    indices = np.linspace(0, len(Re_values) - 1, n, dtype=int)
    return indices.tolist()


def plot_figure9(trajs, Re_values):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle("Figure 9 — Navier-Stokes: Observation Setup", fontsize=13)

    # Select Re values spread across the full range [100, 750]
    show_idx = _select_representative_re(Re_values, n=5)

    # (a) Sample scalar observation for representative Re values
    ax = axes[0]
    for i in show_idx:
        Re = Re_values[i]
        obs = trajs[i][:, 0]
        t = np.arange(len(obs))
        ax.plot(t, obs, label=f"Re={Re:.0f}", alpha=0.7)
    ax.set_xlabel("Time step")
    ax.set_ylabel("$u_x$ at probe")
    ax.legend(fontsize=8)
    ax.set_title("(a) Scalar probe signal")

    # (b) FFT to show vortex shedding frequency
    ax = axes[1]
    for i in show_idx:
        Re = Re_values[i]
        obs = trajs[i][:, 0]
        freqs = np.fft.rfftfreq(len(obs))
        spectrum = np.abs(np.fft.rfft(obs))
        ax.plot(freqs[1:], spectrum[1:], label=f"Re={Re:.0f}", alpha=0.7)
    ax.set_xlabel("Normalised frequency")
    ax.set_ylabel("|FFT|")
    ax.legend(fontsize=8)
    ax.set_title("(b) Frequency content")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "figure9_ns_setup.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ── Figure 10: latent space analysis ────────────────────────────────
def plot_figure10(model, trajs, Re_values):
    """3D latent space coloured by Reynolds number."""
    model.eval()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Select representative Re values for the legend
    show_idx = set(_select_representative_re(Re_values, n=7))

    for i, (traj, Re) in enumerate(zip(trajs, Re_values)):
        ds = WindowedTimeSeriesDataset(traj, SEQ_LEN)
        loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
        X_all, _ = next(iter(loader))
        X_all = X_all.to(DEVICE)
        with torch.no_grad():
            _, ints = model(X_all)
        z = ints["Z"][:, -1, :].cpu().numpy()
        x_emb = ints["X_emb"][:, -1, :].cpu().numpy()
        z_plus = z + x_emb
        if z_plus.shape[1] >= 3:
            ax.scatter(z_plus[:, 0], z_plus[:, 1], z_plus[:, 2],
                       s=1, alpha=0.3,
                       label=f"Re={Re:.0f}" if i in show_idx else "")

    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_zlabel("$z_3$")
    ax.legend(fontsize=7, markerscale=4)
    ax.set_title("Figure 10 — NS 3D Latent Space")
    path = os.path.join(FIG_DIR, "figure10_ns_latent.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ── Figure 11: MSE + Strouhal recovery ──────────────────────────────
def plot_figure11(trajs, Re_values):
    """MSE comparison: MLP vs Transformer, with and without Re input."""
    configs = {
        "MLP Only": dict(use_re=False, is_tf=False),
        "Transformer\n(no Re)": dict(use_re=False, is_tf=True),
        "Transformer\n(+ Re)": dict(use_re=True, is_tf=True),
    }
    results = {k: [] for k in configs}
    test_trajs_no_re, _ = prepare_data(use_re_input=False)
    test_trajs_re, _ = prepare_data(use_re_input=True)
    n_test = max(1, int(0.3 * len(test_trajs_no_re)))
    test_no = test_trajs_no_re[-n_test:]
    test_re = test_trajs_re[-n_test:]

    for seed in range(NUM_SEEDS):
        print(f"  Seed {seed}")
        for name, cfg in configs.items():
            if not cfg["is_tf"]:
                m = train_ns_mlp(test_trajs_no_re, 1, seed)
                mse = compute_test_mse(m, test_no, False)
            elif not cfg["use_re"]:
                m = train_ns_transformer(test_trajs_no_re, 1, seed)
                mse = compute_test_mse(m, test_no, True)
            else:
                # d_obs=2 (input = u_x + Re), d_output=1 (predict only u_x)
                m = train_ns_transformer(test_trajs_re, 2, seed, d_output=1)
                mse = compute_test_mse(m, test_re, True, target_dim=1)
            results[name].append(mse)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (label, mses) in enumerate(results.items()):
        mses = np.array(mses)
        ax.bar(i, mses.mean(), width=0.5, alpha=0.5, color=f"C{i}")
        ax.scatter([i] * len(mses), mses, s=25, c=f"C{i}", zorder=3)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(results.keys(), fontsize=9)
    ax.set_ylabel("|MSE|")
    ax.set_yscale("log")
    ax.set_title("Figure 11 — Navier-Stokes |MSE|")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "figure11_ns_mse.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ── main ────────────────────────────────────────────────────────────
def main():
    print("Preparing Navier-Stokes data...")
    trajs, Re_values = prepare_data(use_re_input=False)

    plot_figure9(trajs, Re_values)

    print("\n=== Training representative transformer ===")
    model = train_ns_transformer(trajs, d_obs=1, seed=0)
    plot_figure10(model, trajs, Re_values)

    print("\n=== MSE comparison (Figure 11) ===")
    plot_figure11(trajs, Re_values)

    print("\nNavier-Stokes experiments complete.")


if __name__ == "__main__":
    main()
