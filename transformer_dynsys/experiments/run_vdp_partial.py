"""
Re-run ONLY the partial observation section of Van der Pol experiments
with more epochs (1000) to ensure convergence.

Full observation (Figure 4b) was fine with 300 epochs — skip it here.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from utils.helpers import set_seed, get_device
from data.vanderpol import generate_vanderpol_data, generate_vanderpol_test_trajectory
from models.transformer import SingleLayerTransformer
from models.mlp import MLPBaseline
from training.dataset import WindowedTimeSeriesDataset, MLPDataset
from training.trainer import Trainer

DEVICE = get_device()
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

MU = 0.5
SEQ_LEN = 5
EPOCHS = 500
LR = 1e-3
BATCH_SIZE = 512
MLP_HIDDEN = 64
NUM_SEEDS = 10


def prepare_data():
    trajs, t_eval = generate_vanderpol_data(
        mu=MU, n_trajectories=300, t_end=6.5, dt=0.1, seed=42
    )
    test_traj, t_test = generate_vanderpol_test_trajectory(mu=MU)
    rng = np.random.default_rng(123)
    indices = rng.permutation(len(trajs))
    trajs_shuffled = [trajs[i] for i in indices]
    n = len(trajs_shuffled)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_trajs = trajs_shuffled[:n_train]
    val_trajs = trajs_shuffled[n_train : n_train + n_val]
    return train_trajs, val_trajs, test_traj, t_test


def _make_partial(trajs, d_obs):
    if d_obs < 2:
        return [tr[:, :d_obs] for tr in trajs]
    return trajs


def compute_test_mse(model, test_traj, d_obs, is_transformer=True):
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


def main():
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    print("Preparing data...")
    train_trajs, val_trajs, test_traj, t_test = prepare_data()

    partial_train = _make_partial(train_trajs, 1)
    partial_val = _make_partial(val_trajs, 1)

    # ── Figure 4c: partial observation MSE ──────────────────────────
    print("\n=== Partial observation experiments (1000 epochs) ===")
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

    # Plot Figure 4c
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, mses) in enumerate(partial_results.items()):
        mses = np.array(mses)
        ax.bar(i, mses.mean(), width=0.5, alpha=0.5, color=f"C{i}")
        ax.scatter([i] * len(mses), mses, s=20, c="black", zorder=3)
        print(f"  {label.replace(chr(10), ' ')}: mean={mses.mean():.6f}, std={mses.std():.6f}")
    ax.set_xticks(range(len(partial_results)))
    ax.set_xticklabels(partial_results.keys(), rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("|MSE|")
    ax.set_yscale("log")
    ax.set_title("Figure 4c — Partial Observation |MSE|")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "figure4c_vdp_partial_mse.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")

    # ── Figure 5: latent analysis ───────────────────────────────────
    print("\n=== Latent analysis (Figure 5) ===")
    from experiments.exp_vanderpol import plot_figure5, ar_predict, collect_internals, extract_attention_patterns
    models_fig5 = {}
    for d_inner, pe in [(1, False), (1, True), (2, False), (2, True)]:
        label = f"Attn {d_inner}D {'PE' if pe else 'NoPE'}"
        m = train_transformer(partial_train, partial_val, 1, d_inner, 1,
                              pe, True, seed=0)
        models_fig5[label] = (m, 1, d_inner, pe)

    plot_figure5(models_fig5, test_traj)
    print("\nPartial observation experiments complete.")


if __name__ == "__main__":
    main()
