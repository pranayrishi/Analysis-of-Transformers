"""
AR spectrum computation, FFT, and spectrograms (Section 3, Figures 2-3).

Key formulas
------------
Scalar AR(p):  H(f) = 1 / |1 - sum_k c_k e^{-j 2pi f k / f_s}|
Vector AR(p):  H(f) = [I - sum_k B_k e^{-j 2pi f k / f_s}]^{-1}
"""

import numpy as np
import torch
from scipy.signal import stft as scipy_stft
from scipy.linalg import expm


# ── scalar AR spectrum ──────────────────────────────────────────────
def compute_ar_spectrum(
    coefficients: np.ndarray, fs: float, n_freqs: int = 2048
):
    """Magnitude of the scalar AR transfer function.

    Parameters
    ----------
    coefficients : array [c_1, c_2, ..., c_p]
        AR coefficients in lag order:  x_t = c_1 x_{t-1} + c_2 x_{t-2} + ...
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    freqs, H_mag : 1-D arrays of length *n_freqs*.
    """
    freqs = np.linspace(0, fs / 2, n_freqs)
    H = np.ones(n_freqs, dtype=complex)
    for k in range(len(coefficients)):
        H -= coefficients[k] * np.exp(-1j * 2 * np.pi * freqs * (k + 1) / fs)
    return freqs, 1.0 / np.abs(H)


# ── vector AR spectrum ──────────────────────────────────────────────
def compute_vector_ar_spectrum(
    B_matrices: list[np.ndarray], fs: float, n_freqs: int = 2048
):
    """Frobenius norm of the vector AR transfer function.

    Parameters
    ----------
    B_matrices : list of (d, d) arrays
        B_k for k = 1 ... p  (lag-1, lag-2, ...).
    """
    freqs = np.linspace(0, fs / 2, n_freqs)
    d = B_matrices[0].shape[0]
    H_norm = np.zeros(n_freqs)

    for fi, f in enumerate(freqs):
        M = np.eye(d, dtype=complex)
        for k, Bk in enumerate(B_matrices):
            M -= Bk * np.exp(-1j * 2 * np.pi * f * (k + 1) / fs)
        try:
            H_norm[fi] = np.linalg.norm(np.linalg.inv(M), "fro")
        except np.linalg.LinAlgError:
            H_norm[fi] = np.inf

    return freqs, H_norm


# ── STFT spectrogram ────────────────────────────────────────────────
def compute_spectrogram(
    signal: np.ndarray, fs: float, nperseg: int = 64, noverlap: int = 48
):
    """Short-time Fourier transform spectrogram."""
    f, t, Zxx = scipy_stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return t, f, np.abs(Zxx)


# ═══════════════════════════════════════════════════════════════════
#  LEARNED AR spectrum extraction (Figures 2 & 3)
# ═══════════════════════════════════════════════════════════════════

def compute_learned_ar_spectrum_sdof(
    model, X_sample, fs: float = 25.0, n_freqs: int = 2048, device="cpu"
):
    """Compute the AR spectrum from a trained attention-only transformer
    for the SCALAR (1-D) SDOF case.

    The effective model is  x_hat = sum_i beta_i * x_tilde_i + const
    where  beta_i = alpha_{n,i} * v_eff  and  v_eff = W_O @ W_V.

    The AR spectrum is  |H(f)| = 1 / |1 - sum_i beta_i exp(-j 2pi f i/fs)|.

    Returns
    -------
    freqs   : (n_freqs,)
    H_mag   : (n_freqs,)
    beta    : (seq_len,)  effective AR coefficients (lag order)
    """
    model.eval()
    model = model.to(device)
    X_sample = X_sample.to(device)

    with torch.no_grad():
        _, internals = model(X_sample)
        A = internals["A"]  # (1, seq_len, seq_len)

    alpha = A[0, -1, :].cpu().numpy()  # attention weights for last query

    # W_V.weight shape: (d_v, d_input);  W_O.weight shape: (d_output, d_v)
    W_V = model.attention.W_V.weight.detach().cpu().numpy()
    if hasattr(model.output_head, "weight"):
        W_O = model.output_head.weight.detach().cpu().numpy()
    else:
        for layer in model.output_head.modules():
            if isinstance(layer, torch.nn.Linear):
                W_O = layer.weight.detach().cpu().numpy()
                break

    M = W_O @ W_V  # (d_output, d_input) — scalar case: (1,1)
    M_scalar = float(M.flat[0])

    # beta_i = alpha_i * M_scalar  (token ordering: 0=oldest, -1=newest)
    beta_token_order = alpha * M_scalar

    # Convert to AR lag order: ar_coeffs[0]=lag-1 (newest), [1]=lag-2, ...
    ar_coeffs = beta_token_order[::-1].copy()

    freqs = np.linspace(0, fs / 2, n_freqs)
    H = np.ones(n_freqs, dtype=complex)
    for k in range(len(ar_coeffs)):
        H -= ar_coeffs[k] * np.exp(-1j * 2 * np.pi * freqs * (k + 1) / fs)
    H_mag = 1.0 / np.abs(H)

    return freqs, H_mag, ar_coeffs


def compute_learned_ar_spectrum_2dof(
    model, X_sample, fs: float = 25.0, n_freqs: int = 2048, device="cpu"
):
    """Compute AR spectrum from a trained attention-only transformer
    for the VECTOR (multi-DOF) case.

    B_i = alpha_{n,i} * M  where M = W_O @ W_V are MATRICES.
    H(f) = [I - sum_k B_k exp(-j 2pi f k / fs)]^{-1}
    Plot ||H(f)||_F.

    Returns
    -------
    freqs   : (n_freqs,)
    H_mag   : (n_freqs,)
    """
    model.eval()
    model = model.to(device)
    X_sample = X_sample.to(device)

    with torch.no_grad():
        _, internals = model(X_sample)
        A = internals["A"]

    alpha = A[0, -1, :].cpu().numpy()
    seq_len = len(alpha)

    W_V = model.attention.W_V.weight.detach().cpu().numpy()
    if hasattr(model.output_head, "weight"):
        W_O = model.output_head.weight.detach().cpu().numpy()
    else:
        for layer in model.output_head.modules():
            if isinstance(layer, torch.nn.Linear):
                W_O = layer.weight.detach().cpu().numpy()
                break

    M = W_O @ W_V  # (d_output, d_input)
    d = M.shape[0]

    # Token-order → lag-order (reverse)
    B_list = [alpha[seq_len - 1 - k] * M for k in range(seq_len)]

    freqs = np.linspace(0, fs / 2, n_freqs)
    H_mag = np.zeros(n_freqs)
    I_mat = np.eye(d)

    for fi, f in enumerate(freqs):
        H_inv = I_mat.copy().astype(complex)
        for k in range(seq_len):
            H_inv -= B_list[k] * np.exp(-1j * 2 * np.pi * f * (k + 1) / fs)
        try:
            H_mag[fi] = np.linalg.norm(np.linalg.inv(H_inv), "fro")
        except np.linalg.LinAlgError:
            H_mag[fi] = np.inf

    return freqs, H_mag


# ── exact AR(2) spectrum from physical parameters ───────────────────
def compute_exact_ar2_spectrum(
    m: float, c: float, k: float, dt: float, fs: float, n_freqs: int = 2048
):
    """Exact AR(2) spectrum from SDOF system parameters.

    x_{t+1} = c1 x_t + c2 x_{t-1}
    where c1 = trace(Phi), c2 = -det(Phi), Phi = expm(A dt).

    Returns
    -------
    freqs, H_mag, ar_coeffs
    """
    A_cont = np.array([[0.0, 1.0], [-k / m, -c / m]])
    Phi = expm(A_cont * dt)
    c1 = np.trace(Phi)
    c2 = -np.linalg.det(Phi)
    ar_coeffs = [c1, c2]

    freqs = np.linspace(0, fs / 2, n_freqs)
    H = np.ones(n_freqs, dtype=complex)
    for idx, coeff in enumerate(ar_coeffs):
        H -= coeff * np.exp(-1j * 2 * np.pi * freqs * (idx + 1) / fs)
    H_mag = 1.0 / np.abs(H)

    return freqs, H_mag, ar_coeffs


# ── legacy wrapper (kept for backward compat) ──────────────────────
def extract_ar_coefficients_from_transformer(model, X_input, device="cpu"):
    """Legacy extraction — prefer the dedicated _sdof / _2dof helpers."""
    model.eval()
    X_input = torch.as_tensor(X_input, dtype=torch.float32).to(device)
    if X_input.dim() == 2:
        X_input = X_input.unsqueeze(0)

    with torch.no_grad():
        _, internals = model(X_input)
        A = internals["A"]

    alpha = A[0, -1, :].cpu().numpy()

    W_V = model.attention.W_V.weight.detach().cpu().numpy()
    if hasattr(model.output_head, "weight"):
        W_O = model.output_head.weight.detach().cpu().numpy()
    else:
        return alpha, alpha

    M = W_O @ W_V
    if M.size == 1:
        m_scalar = float(M.flat[0])
        effective_coeffs = [float(alpha[i]) * m_scalar for i in range(len(alpha))]
    else:
        effective_coeffs = [alpha[i] * M for i in range(len(alpha))]
    return alpha, effective_coeffs
