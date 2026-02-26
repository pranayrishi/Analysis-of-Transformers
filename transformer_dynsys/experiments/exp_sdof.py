"""
Section 3.1 — SDOF structural system experiments.

Reproduces **Figure 2**: time domain, AR spectrum, and spectrogram for
  Case 1 (k = 2000, both AR coefficients negative -> attention succeeds)
  Case 2 (k =  500, mixed-sign AR coefficients   -> attention fails)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.helpers import set_seed, get_device
from data.sdof import generate_sdof_data, compute_sdof_ar2_coefficients
from models.transformer import SingleLayerTransformer
from training.dataset import WindowedTimeSeriesDataset
from training.trainer import Trainer
from analysis.spectral import (
    compute_learned_ar_spectrum_sdof,
    compute_exact_ar2_spectrum,
    compute_spectrogram,
)

DEVICE = get_device()
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── hyperparameters (Section 3.1) ──────────────────────────────────
FS = 25.0
DT = 1.0 / FS
SEQ_LEN = 2        # AR(2): two past observations
D_OBS = 1
D_INNER = 1
EPOCHS = 2000
LR = 1e-3
BATCH_SIZE = 64


# ── autoregressive roll-out ─────────────────────────────────────────
def autoregressive_predict(model, initial_window, n_steps, device=DEVICE):
    model.eval()
    window = initial_window.copy()
    preds = []
    with torch.no_grad():
        for _ in range(n_steps):
            X = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            y, _ = model(X)
            y_np = y.cpu().numpy().squeeze()
            preds.append(float(y_np))
            window = np.roll(window, -1, axis=0)
            window[-1] = float(y_np)
    return np.array(preds)


# ── run one case ────────────────────────────────────────────────────
def run_sdof_case(k, case_label, seed=0):
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"SDOF Case: k={k} N/m  ({case_label})")
    print(f"{'='*60}")

    # — data —
    t, x, v = generate_sdof_data(k=k, t_end=10.0, fs=FS)
    x_2d = x[:, None]  # (T, 1)

    # analytical AR(2) coefficients
    c1, c2 = compute_sdof_ar2_coefficients(m=1.0, c=0.5, k=k, dt=DT)
    fn = (1 / (2 * np.pi)) * np.sqrt(k / 1.0)
    print(f"  Exact AR(2): c1 = {c1:.4f},  c2 = {c2:.4f}")
    print(f"  Natural frequency: {fn:.2f} Hz")

    # — dataset —
    dataset = WindowedTimeSeriesDataset(x_2d, window_size=SEQ_LEN)
    n_train = int(0.8 * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, range(n_train))
    val_ds = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))

    # — model (attention-only, linear output, no residual) —
    model = SingleLayerTransformer(
        d_obs=D_OBS, d_inner=D_INNER, d_k=D_INNER, d_v=D_INNER,
        d_output=D_OBS, seq_len=SEQ_LEN,
        use_mlp=False,
        use_positional_encoding=True,
        use_residual=False,
    )

    # — train —
    trainer = Trainer(
        model, train_ds, val_ds,
        lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS,
        device=DEVICE, verbose=True,
    )
    model = trainer.train()

    # — LEARNED AR spectrum (corrected: uses alpha * M) —
    sample_input = torch.tensor(x_2d[:SEQ_LEN][None], dtype=torch.float32)
    freqs_learned, H_learned, beta = compute_learned_ar_spectrum_sdof(
        model, sample_input, fs=FS, device=DEVICE,
    )
    print(f"  Effective AR coefficients (lag order): {beta}")

    # — EXACT AR(2) spectrum from physical parameters —
    freqs_exact, H_exact, ar_exact = compute_exact_ar2_spectrum(
        m=1.0, c=0.5, k=k, dt=DT, fs=FS,
    )
    print(f"  Exact AR(2) coefficients: {ar_exact}")

    # — autoregressive prediction —
    init_window = x_2d[:SEQ_LEN]
    n_pred = len(x) - SEQ_LEN
    preds = autoregressive_predict(model, init_window, n_pred, DEVICE)

    # — spectrogram of prediction —
    nperseg = min(64, len(preds) // 2)
    noverlap = nperseg * 3 // 4
    t_stft, f_stft, Sxx = compute_spectrogram(
        preds, FS, nperseg=nperseg, noverlap=noverlap
    )

    return dict(
        t=t, x=x, preds=preds, fn=fn, c1=c1, c2=c2,
        freqs_learned=freqs_learned, H_learned=H_learned,
        freqs_exact=freqs_exact, H_exact=H_exact,
        t_stft=t_stft, f_stft=f_stft, Sxx=Sxx,
        case_label=case_label, beta=beta,
    )


# ── Figure 2 ────────────────────────────────────────────────────────
def plot_figure2(results_list):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Figure 2 — SDOF System: Attention-Only Transformer", fontsize=14)

    for row, res in enumerate(results_list):
        t = res["t"]
        x = res["x"]
        preds = res["preds"]
        fn = res["fn"]

        # (a) time domain
        ax = axes[row, 0]
        ax.plot(t, x, "k-", label="True", linewidth=1)
        t_pred = t[SEQ_LEN : SEQ_LEN + len(preds)]
        ax.plot(t_pred, preds[:len(t_pred)], "r--", label="Predicted", linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Displacement (m)")
        ax.set_title(f"{res['case_label']}: Time Domain")
        ax.legend()

        # (b) AR spectrum — THE KEY PLOT
        ax = axes[row, 1]
        ax.plot(res["freqs_learned"], res["H_learned"], "b-",
                linewidth=1.5, label="Learned coefficients")
        ax.plot(res["freqs_exact"], res["H_exact"], "g--",
                linewidth=1.5, alpha=0.7, label="Exact AR(2)")
        ax.axvline(fn, color="r", ls="--", alpha=0.7,
                   label=f"$f_n$ = {fn:.2f} Hz")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("$|H(e^{j\\omega})|$")
        ax.set_title(f"{res['case_label']}: AR Spectrum")
        ax.legend(fontsize=8)

        # (c) spectrogram
        ax = axes[row, 2]
        if res["Sxx"].size > 0:
            im = ax.pcolormesh(res["t_stft"], res["f_stft"],
                               res["Sxx"].squeeze(), shading="gouraud",
                               cmap="viridis")
            fig.colorbar(im, ax=ax, label="$|H|$")
            ax.axhline(fn, color="r", ls="--", alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"{res['case_label']}: Spectrogram")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "figure2_sdof.png")
    plt.savefig(path)
    plt.close()
    print(f"\nSaved: {path}")


# ── main ────────────────────────────────────────────────────────────
def main():
    results = []
    for k, label in [(2000.0, "Case 1 (k=2000)"), (500.0, "Case 2 (k=500)")]:
        res = run_sdof_case(k, label, seed=0)
        results.append(res)
    plot_figure2(results)


if __name__ == "__main__":
    main()
