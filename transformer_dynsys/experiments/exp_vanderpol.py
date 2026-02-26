"""
Section 4.1 — Van der Pol oscillator experiments.

Reproduces **Figures 4-5**:
  Fig 4a: Phase portrait (train=black / val=green / test=red)
  Fig 4b: Full-observation |MSE| across 10 seeds
  Fig 4c: Partial-observation |MSE| across 10 seeds
  Fig 5a: Predicted time series on limit cycle
  Fig 5b: Z (pure attention) vs x
  Fig 5c: Z + x (residual output) vs x
  Fig 5d: 2D latent trajectories (Z_residual)
  Fig 5e-g: Attention heatmaps across seeds at 3 time points
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from utils.helpers import set_seed, get_device
from data.vanderpol import generate_vanderpol_data, generate_vanderpol_test_trajectory
from models.transformer import SingleLayerTransformer
from models.mlp import MLPBaseline
from training.dataset import WindowedTimeSeriesDataset, MLPDataset
from training.trainer import Trainer

DEVICE = get_device()
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── constants ───────────────────────────────────────────────────────
MU = 0.5
SEQ_LEN = 5
EPOCHS = 300
LR = 1e-3
BATCH_SIZE = 64
MLP_HIDDEN = 64
NUM_SEEDS = 10


# ── data preparation ────────────────────────────────────────────────
def prepare_data():
    trajs, t_eval = generate_vanderpol_data(
        mu=MU, n_trajectories=300, t_end=6.5, dt=0.1, seed=42
    )
    test_traj, t_test = generate_vanderpol_test_trajectory(mu=MU)

    # Shuffle with fixed seed for reproducibility
    rng = np.random.default_rng(123)
    indices = rng.permutation(len(trajs))
    trajs_shuffled = [trajs[i] for i in indices]

    n = len(trajs_shuffled)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_trajs = trajs_shuffled[:n_train]
    val_trajs = trajs_shuffled[n_train : n_train + n_val]

    return train_trajs, val_trajs, test_traj, t_test


# ── single-step test MSE (NOT autoregressive) ──────────────────────
def compute_test_mse(model, test_traj, d_obs, is_transformer=True):
    """One-step-ahead MSE on the test limit-cycle trajectory."""
    model.eval()
    obs = test_traj[:, :d_obs] if d_obs < test_traj.shape[1] else test_traj

    all_preds, all_tgts = [], []
    with torch.no_grad():
        if is_transformer:
            for t in range(len(obs) - SEQ_LEN):
                window = obs[t : t + SEQ_LEN]
                target = obs[t + SEQ_LEN]
                X = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                y, _ = model(X)
                all_preds.append(y.cpu().numpy().squeeze())
                all_tgts.append(target)
        else:
            for t in range(len(obs) - 1):
                X = torch.tensor(obs[t], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                y = model(X)
                all_preds.append(y.cpu().numpy().squeeze())
                all_tgts.append(obs[t + 1])

    preds = np.array(all_preds).flatten()
    tgts = np.array(all_tgts).flatten()
    return float(np.mean((preds - tgts) ** 2))


# ── autoregressive rollout (for Figure 5a) ─────────────────────────
def ar_predict(model, init_window, n_steps):
    model.eval()
    window = init_window.copy()
    preds = []
    with torch.no_grad():
        for _ in range(n_steps):
            X = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            y, _ = model(X)
            y_np = y.cpu().numpy().squeeze()
            preds.append(y_np.copy() if y_np.ndim > 0 else np.array([y_np]))
            window = np.roll(window, -1, axis=0)
            window[-1] = y_np if y_np.ndim > 0 else [float(y_np)]
    return np.array(preds)


# ── training wrappers ───────────────────────────────────────────────
def _make_partial(trajs, d_obs):
    if d_obs < 2:
        return [tr[:, :d_obs] for tr in trajs]
    return trajs


def train_transformer(train_trajs, val_trajs, d_obs, d_inner, d_output,
                      use_pe, use_residual, seed):
    set_seed(seed)
    train_ds = WindowedTimeSeriesDataset(train_trajs, SEQ_LEN,
                                          target_dim=d_output if d_output < 2 else None)
    val_ds = WindowedTimeSeriesDataset(val_trajs, SEQ_LEN,
                                        target_dim=d_output if d_output < 2 else None)
    model = SingleLayerTransformer(
        d_obs=d_obs, d_inner=d_inner, d_output=d_output, seq_len=SEQ_LEN,
        use_mlp=True, mlp_hidden=MLP_HIDDEN,
        use_positional_encoding=use_pe, use_residual=use_residual,
    )
    trainer = Trainer(model, train_ds, val_ds,
                      lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS,
                      device=DEVICE, verbose=False)
    return trainer.train()


def train_mlp(train_trajs, val_trajs, d_input, d_output, seed):
    set_seed(seed)
    train_ds = MLPDataset(train_trajs,
                           input_dim=d_input if d_input < 2 else None,
                           target_dim=d_output if d_output < 2 else None)
    val_ds = MLPDataset(val_trajs,
                         input_dim=d_input if d_input < 2 else None,
                         target_dim=d_output if d_output < 2 else None)
    model = MLPBaseline(d_input, d_output, hidden_size=MLP_HIDDEN)
    trainer = Trainer(model, train_ds, val_ds,
                      lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS,
                      device=DEVICE, verbose=False)
    return trainer.train()


# ── Figure 4a: phase portrait ──────────────────────────────────────
def plot_figure4a(train_trajs, val_trajs, test_traj):
    fig, ax = plt.subplots(figsize=(6, 6))

    for tr in train_trajs[:80]:
        ax.plot(tr[:, 0], tr[:, 1], color="black", alpha=0.08, linewidth=0.5)
    for tr in val_trajs[:30]:
        ax.plot(tr[:, 0], tr[:, 1], color="green", alpha=0.15, linewidth=0.5)
    ax.plot(test_traj[:, 0], test_traj[:, 1], color="red", linewidth=2.0,
            label="Test (limit cycle)")

    # Dummy legend entries
    ax.plot([], [], color="black", linewidth=1.5, label="Train")
    ax.plot([], [], color="green", linewidth=1.5, label="Validation")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$\\dot{x}$")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.set_title("Figure 4a — Van der Pol Phase Portrait")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "figure4a_vdp_phase.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ── MSE bar plot ────────────────────────────────────────────────────
def plot_mse_bar(results, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, mses) in enumerate(results.items()):
        mses = np.array(mses)
        ax.bar(i, mses.mean(), width=0.5, alpha=0.5, color=f"C{i}")
        ax.scatter([i] * len(mses), mses, s=20, c="black", zorder=3)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(results.keys(), rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("|MSE|")
    ax.set_yscale("log")
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ── attention pattern extraction ────────────────────────────────────
def extract_attention_patterns(model, test_obs, time_indices=(10, 30, 60)):
    """Extract last-row attention at specific time indices on the test traj."""
    model.eval()
    patterns = {}
    with torch.no_grad():
        for ti in time_indices:
            if ti < SEQ_LEN:
                continue
            window = test_obs[ti - SEQ_LEN : ti]
            X = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            _, ints = model(X)
            patterns[ti] = ints["A"][0, -1, :].cpu().numpy()
    return patterns


# ── collect internals over test traj ────────────────────────────────
def collect_internals(model, test_traj, d_obs):
    obs = test_traj[:, :d_obs] if d_obs < test_traj.shape[1] else test_traj
    ds = WindowedTimeSeriesDataset(obs, SEQ_LEN)
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    X_all, y_all = next(iter(loader))
    X_all = X_all.to(DEVICE)
    model.eval()
    with torch.no_grad():
        _, ints = model(X_all)
    return {k: v.cpu().numpy() for k, v in ints.items()}, X_all.cpu().numpy()


# ── Figure 5 ────────────────────────────────────────────────────────
def plot_figure5(models_dict, test_traj):
    t_test = np.arange(len(test_traj)) * 0.1
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Figure 5 — Van der Pol Latent Analysis", fontsize=14)

    # 5a: predicted time series (partial obs, 1D latent)
    ax = fig.add_subplot(3, 3, 1)
    true_x = test_traj[:, 0]
    n_show = 200
    ax.plot(t_test[:n_show], true_x[:n_show], "ko", ms=2, label="Truth")
    for label, (model, d_obs, d_inner, pe) in models_dict.items():
        if d_inner == 1 and d_obs == 1:
            init = test_traj[:SEQ_LEN, :1]
            preds = ar_predict(model, init, n_show - SEQ_LEN)
            color = "r" if not pe else "c"
            ax.plot(t_test[SEQ_LEN:n_show], preds[:n_show - SEQ_LEN, 0],
                    color=color, ms=3, alpha=0.6, label=label, linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("x")
    ax.set_title("(a) Predicted time series")
    ax.legend(fontsize=7)

    # 5b: Z (pure attention, BEFORE residual) vs x
    ax = fig.add_subplot(3, 3, 2)
    for label, (model, d_obs, d_inner, pe) in models_dict.items():
        if d_inner == 1 and d_obs == 1:
            ints, X_all = collect_internals(model, test_traj, d_obs)
            z_pure = ints["Z"][:, -1, 0]       # pure attention (pre-residual)
            x_last = X_all[:, -1, 0]
            color = "r" if not pe else "c"
            ax.scatter(x_last, z_pure, s=1, alpha=0.3, c=color, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("Z")
    ax.set_title("(b) Z vs x")
    ax.legend(fontsize=7, markerscale=4)

    # 5c: Z + x (residual output) vs x
    ax = fig.add_subplot(3, 3, 3)
    for label, (model, d_obs, d_inner, pe) in models_dict.items():
        if d_inner == 1 and d_obs == 1:
            ints, X_all = collect_internals(model, test_traj, d_obs)
            z_res = ints["Z_residual"][:, -1, 0]  # post-residual
            x_last = X_all[:, -1, 0]
            color = "r" if not pe else "c"
            ax.scatter(x_last, z_res, s=1, alpha=0.3, c=color, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("Z + x")
    ax.set_title("(c) Z + x vs x")
    ax.legend(fontsize=7, markerscale=4)

    # 5d: 2D latent trajectories (Z_residual)
    ax = fig.add_subplot(3, 3, 4)
    for label, (model, d_obs, d_inner, pe) in models_dict.items():
        if d_inner == 2 and d_obs == 1:
            ints, _ = collect_internals(model, test_traj, d_obs)
            z_res = ints["Z_residual"][:, -1, :]  # (N, 2)
            color = "r" if not pe else "c"
            ax.scatter(z_res[:, 0], z_res[:, 1], s=1, alpha=0.3,
                       c=color, label=label)
    ax.set_xlabel("$z_1 + x^{emb}_1$")
    ax.set_ylabel("$z_2 + x^{emb}_2$")
    ax.set_title("(d) 2D Latent")
    ax.legend(fontsize=7, markerscale=4)

    # 5e-g: attention heatmaps across seeds × time points
    configs = [
        ("(e) 1D+PE", 1, 1, True),
        ("(f) 2D+PE", 1, 2, True),
        ("(g) 2D NoPE", 1, 2, False),
    ]
    time_indices = (10, 30, 60)

    for ci, (panel_title, d_obs_c, d_inner_c, pe_c) in enumerate(configs):
        ax = fig.add_subplot(3, 3, 7 + ci)
        rows = []
        for label, (model, d_obs, d_inner, pe) in models_dict.items():
            if d_inner == d_inner_c and d_obs == d_obs_c and pe == pe_c:
                obs = test_traj[:, :d_obs]
                pats = extract_attention_patterns(model, obs, time_indices)
                for ti in time_indices:
                    if ti in pats:
                        rows.append(pats[ti])
        if rows:
            mat = np.array(rows)
            im = ax.imshow(mat, aspect="auto", cmap="hot", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(mat.shape[1]))
            ax.set_xlabel("Token")
            ax.set_ylabel("Seed / time")
        ax.set_title(panel_title)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "figure5_vdp_latent.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ── main ────────────────────────────────────────────────────────────
def main():
    print("Generating Van der Pol data...")
    train_trajs, val_trajs, test_traj, t_test = prepare_data()
    plot_figure4a(train_trajs, val_trajs, test_traj)

    # ── Figure 4b: full observation ─────────────────────────────────
    print("\n=== Full observation experiments ===")
    full_results = {
        "MLP Only": [],
        "MLP+Attn 2D\n(No P.E.)": [],
        "MLP+Attn 2D\n(P.E.)": [],
    }
    for seed in range(NUM_SEEDS):
        print(f"  Seed {seed}")
        m = train_mlp(train_trajs, val_trajs, 2, 2, seed)
        full_results["MLP Only"].append(compute_test_mse(m, test_traj, 2, False))

        m = train_transformer(train_trajs, val_trajs, 2, 2, 2, False, True, seed)
        full_results["MLP+Attn 2D\n(No P.E.)"].append(
            compute_test_mse(m, test_traj, 2, True))

        m = train_transformer(train_trajs, val_trajs, 2, 2, 2, True, True, seed)
        full_results["MLP+Attn 2D\n(P.E.)"].append(
            compute_test_mse(m, test_traj, 2, True))

    plot_mse_bar(full_results,
                 "Figure 4b — Full Observation |MSE|",
                 "figure4b_vdp_full_mse.png")

    # ── Figure 4c: partial observation ──────────────────────────────
    print("\n=== Partial observation experiments ===")
    partial_train = _make_partial(train_trajs, 1)
    partial_val = _make_partial(val_trajs, 1)

    partial_results = {
        "MLP Only": [],
        "MLP+Attn 2D\n(No P.E.)": [],
        "MLP+Attn 3D\n(No P.E.)": [],
        "MLP+Attn 2D\n(P.E.)": [],
        "MLP+Attn 3D\n(P.E.)": [],
    }
    for seed in range(NUM_SEEDS):
        print(f"  Seed {seed}")
        m = train_mlp(partial_train, partial_val, 1, 1, seed)
        partial_results["MLP Only"].append(
            compute_test_mse(m, test_traj, 1, False))

        for d_inner, pe, key in [
            (2, False, "MLP+Attn 2D\n(No P.E.)"),
            (3, False, "MLP+Attn 3D\n(No P.E.)"),
            (2, True,  "MLP+Attn 2D\n(P.E.)"),
            (3, True,  "MLP+Attn 3D\n(P.E.)"),
        ]:
            m = train_transformer(partial_train, partial_val, 1, d_inner, 1,
                                  pe, True, seed)
            partial_results[key].append(
                compute_test_mse(m, test_traj, 1, True))

    plot_mse_bar(partial_results,
                 "Figure 4c — Partial Observation |MSE|",
                 "figure4c_vdp_partial_mse.png")

    # ── Figure 5: latent analysis ───────────────────────────────────
    print("\n=== Latent analysis (Figure 5) ===")
    models_fig5 = {}
    for d_inner, pe in [(1, False), (1, True), (2, False), (2, True)]:
        label = f"Attn {d_inner}D {'PE' if pe else 'NoPE'}"
        m = train_transformer(partial_train, partial_val, 1, d_inner, 1,
                              pe, True, seed=0)
        models_fig5[label] = (m, 1, d_inner, pe)

    plot_figure5(models_fig5, test_traj)
    print("\nVan der Pol experiments complete.")


if __name__ == "__main__":
    main()
