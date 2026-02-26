"""
Section 3.2 — Two-DOF coupled oscillator experiments.

Reproduces **Figure 3**: AR transfer-function spectra for
  (a) Full observation (d_obs=2), seq_len=4   -> both peaks recovered
  (b) Partial observation (d_obs=1), seq_len=4 -> fails (broad hump)
  (c) Partial observation (d_obs=1), seq_len=8 -> succeeds (both peaks)

CRITICAL: the spectrum is the AR *transfer function*
  H(f) = [I - sum_k B_k exp(-j2pi f k/fs)]^{-1}
evaluated from the learned attention weights and projections.
It is NOT an FFT of any signal.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.helpers import set_seed, get_device
from data.mdof import generate_2dof_data
from models.transformer import SingleLayerTransformer
from training.dataset import WindowedTimeSeriesDataset
from training.trainer import Trainer

DEVICE = get_device()
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FS = 25.0
EPOCHS = 2000
LR = 1e-3
BATCH_SIZE = 64


# ── AR transfer function from model weights ─────────────────────────
def compute_2dof_ar_spectrum_from_model(model, X_sample, fs=25.0,
                                         n_freqs=2048, device="cpu"):
    """Compute ||H(f)||_F from learned attention weights + projections.

    H(f) = [I - sum_{k=1}^n B_k exp(-j 2pi f k / fs)]^{-1}
    where B_k = alpha_{n, n+1-k} * M  and  M = W_O @ W_V.

    This is a SMOOTH rational function — NOT an FFT of any signal.
    """
    model.eval()
    X_sample = X_sample.to(device)

    with torch.no_grad():
        _, internals = model(X_sample)
        A = internals["A"]  # (1, seq_len, seq_len)

    alpha = A[0, -1, :].cpu().numpy()   # (seq_len,)
    seq_len = len(alpha)

    W_V = model.attention.W_V.weight.detach().cpu().numpy()   # (d_v, d_in)
    W_O = model.output_head.weight.detach().cpu().numpy()     # (d_out, d_v)
    M = W_O @ W_V                                             # (d_out, d_in)

    d = M.shape[0]

    # B_k for lag k: newest token (index seq_len-1) is lag 1
    B = [alpha[seq_len - k] * M for k in range(1, seq_len + 1)]

    # Diagnostics
    print(f"    alpha  = {alpha}")
    print(f"    M      = {M.flatten()}" if M.size <= 4 else f"    M shape={M.shape}")
    for i, Bk in enumerate(B):
        tag = f"B_{i+1}"
        if Bk.size <= 4:
            print(f"    {tag}     = {Bk.flatten()}")
        else:
            print(f"    {tag} frob = {np.linalg.norm(Bk, 'fro'):.4f}")

    # Evaluate transfer function
    freqs = np.linspace(0, fs / 2, n_freqs)
    H_mag = np.zeros(n_freqs)
    I_mat = np.eye(d, dtype=complex)

    for fi, f in enumerate(freqs):
        H_inv = I_mat.copy()
        for k, Bk in enumerate(B):
            H_inv -= Bk.astype(complex) * np.exp(
                -1j * 2 * np.pi * f * (k + 1) / fs
            )
        try:
            H_mag[fi] = np.linalg.norm(np.linalg.inv(H_inv), "fro")
        except np.linalg.LinAlgError:
            H_mag[fi] = 1e6

    return freqs, H_mag


# ── Averaged AR spectrum across multiple input windows ──────────────
def compute_averaged_ar_spectrum(model, obs, n_samples=50, fs=25.0,
                                  n_freqs=2048, device="cpu"):
    """Compute AR transfer function averaged across multiple input windows.

    Averaging smooths oscillatory artifacts that arise from
    individual attention weight configurations (since alpha depends
    on the specific input through Q·K^T scores).
    """
    model.eval()
    seq_len = model.seq_len

    T = len(obs)
    start_indices = np.linspace(seq_len, T - 1, n_samples, dtype=int)

    W_V = model.attention.W_V.weight.detach().cpu().numpy()
    W_O = model.output_head.weight.detach().cpu().numpy()
    M = W_O @ W_V
    d = M.shape[0]

    all_H_mag = []

    for start in start_indices:
        window = obs[start - seq_len:start]  # (seq_len, d_obs)
        X = torch.tensor(window[None], dtype=torch.float32).to(device)

        with torch.no_grad():
            _, internals = model(X)
            A = internals["A"]

        alpha = A[0, -1, :].cpu().numpy()

        B = [alpha[seq_len - k] * M for k in range(1, seq_len + 1)]

        freqs = np.linspace(0, fs / 2, n_freqs)
        H_mag = np.zeros(n_freqs)
        I_mat = np.eye(d, dtype=complex)

        for fi, f in enumerate(freqs):
            H_inv = I_mat.copy()
            for k_idx, Bk in enumerate(B):
                H_inv -= Bk.astype(complex) * np.exp(
                    -1j * 2 * np.pi * f * (k_idx + 1) / fs
                )
            try:
                H_mag[fi] = np.linalg.norm(np.linalg.inv(H_inv), "fro")
            except np.linalg.LinAlgError:
                H_mag[fi] = 1e6

        all_H_mag.append(H_mag)

    avg_H = np.mean(all_H_mag, axis=0)
    print(f"    Averaged over {len(all_H_mag)} windows")
    return freqs, avg_H


# ── run one case ────────────────────────────────────────────────────
def run_2dof_case(d_obs, seq_len, case_label, seed=0):
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"2DOF: d_obs={d_obs}, seq_len={seq_len}  ({case_label})")
    print(f"{'='*60}")

    t, x1, x2, v1, v2 = generate_2dof_data(t_end=25.0, fs=FS)

    obs = np.column_stack([x1, x2]) if d_obs == 2 else x1[:, None]

    # Analytical natural frequencies
    M_inv_K = np.array([[2500, -1500], [-1500, 1500]])
    eigvals = np.linalg.eigvalsh(M_inv_K)
    f_natural = np.sqrt(eigvals) / (2 * np.pi)
    print(f"  Natural frequencies: {f_natural[0]:.2f}, {f_natural[1]:.2f} Hz")

    d_inner = d_obs

    dataset = WindowedTimeSeriesDataset(obs, window_size=seq_len)
    n_train = int(0.8 * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, range(n_train))
    val_ds = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))

    model = SingleLayerTransformer(
        d_obs=d_obs, d_inner=d_inner, d_k=d_inner, d_v=d_inner,
        d_output=d_obs, seq_len=seq_len,
        use_mlp=False,
        use_positional_encoding=True,
        use_residual=False,
    )

    trainer = Trainer(
        model, train_ds, val_ds,
        lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS,
        device=DEVICE, verbose=True,
    )
    model = trainer.train()

    # Compute AVERAGED AR transfer function across multiple windows
    print("  Extracting averaged AR spectrum from model weights:")
    freqs, H = compute_averaged_ar_spectrum(
        model, obs, n_samples=50, fs=FS, device=DEVICE
    )
    print(f"  H range: [{H.min():.4f}, {H.max():.4f}]")

    return dict(freqs=freqs, H=H, f_natural=f_natural, case_label=case_label)


# ── Figure 3 ────────────────────────────────────────────────────────
def plot_figure3(results_list):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Figure 3 — 2DOF System: AR Transfer-Function Spectra",
                 fontsize=14)

    for ax, res in zip(axes, results_list):
        ax.plot(res["freqs"], res["H"], "b-", linewidth=1.5,
                label="Learned spectrum")
        for fn in res["f_natural"]:
            ax.axvline(fn, color="r", ls="--", alpha=0.7)
        ax.axvline(res["f_natural"][0], color="r", ls="--", alpha=0.0,
                   label="Natural Frequencies")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("$\\|H(e^{j\\omega})\\|_F$")
        ax.set_title(res["case_label"])
        ax.legend(fontsize=8)
        ax.set_xlim(0, FS / 2)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "figure3_2dof.png")
    plt.savefig(path)
    plt.close()
    print(f"\nSaved: {path}")


def main():
    results = []
    results.append(run_2dof_case(d_obs=2, seq_len=4,
                                 case_label="(a) Full obs, n=4"))
    results.append(run_2dof_case(d_obs=1, seq_len=4,
                                 case_label="(b) Partial obs, n=4"))
    results.append(run_2dof_case(d_obs=1, seq_len=8,
                                 case_label="(c) Partial obs, n=8"))
    plot_figure3(results)


if __name__ == "__main__":
    main()
