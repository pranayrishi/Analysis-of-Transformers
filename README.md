# Reproducing: A Mechanistic Analysis of Transformers for Dynamical Systems

A PyTorch reproduction of **"A Mechanistic Analysis of Transformers for Dynamical Systems"** by Duthé et al. ([arXiv:2512.21113v1](https://arxiv.org/abs/2512.21113), December 2025).

This repository provides a complete implementation of the single-layer, single-head Transformer architecture analyzed in the paper, along with experiments reproducing key results on linear and nonlinear dynamical systems.

---

## Overview

The paper investigates how Transformers learn to predict dynamical systems by analyzing the learned attention patterns and connecting them to classical autoregressive (AR) models. Key findings include:

- **Linear systems (SDOF/2DOF):** The Transformer learns AR coefficients that match the theoretical spectral response
- **Nonlinear systems (Van der Pol, Chafee-Infante, Navier-Stokes):** The model discovers meaningful latent representations and dynamics

---

## Project Structure

```
transformer_dynsys/
├── models/
│   ├── transformer.py    # Single-layer Transformer (Figure 1)
│   ├── attention.py      # Causal self-attention mechanism
│   └── mlp.py            # MLP baseline model
├── data/
│   ├── sdof.py           # SDOF oscillator data generation
│   ├── mdof.py           # 2DOF coupled oscillator
│   ├── vanderpol.py      # Van der Pol oscillator
│   ├── chafee_infante.py # Chafee-Infante PDE
│   └── navier_stokes.py  # Navier-Stokes flow
├── training/
│   ├── trainer.py        # Generic training loop
│   └── dataset.py        # PyTorch Dataset utilities
├── analysis/
│   ├── spectral.py       # AR spectrum computation (Figures 2-3)
│   ├── attention_viz.py  # Attention pattern visualization
│   └── latent_viz.py     # Latent space analysis
├── experiments/
│   ├── exp_sdof.py       # Section 3.1 experiments
│   ├── exp_2dof.py       # Section 3.2 experiments
│   ├── exp_vanderpol.py  # Section 4.1 experiments
│   ├── exp_chafee_infante.py  # Section 4.2 experiments
│   └── exp_navier_stokes.py   # Section 4.3 experiments
├── config/
│   └── defaults.py       # Hyperparameters from the paper
├── figures/              # Output directory for plots
└── run_all.py            # Master experiment runner
```

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, SciPy, Matplotlib

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd "Reproducing - A Mechanistic Analysis of Transformers for Dynamical Systems"

# Install dependencies
pip install torch numpy scipy matplotlib
```

---

## Usage

### Run All Experiments

```bash
python transformer_dynsys/run_all.py
```

> **Note:** Running all experiments takes several hours due to training 10 models per configuration.

### Run Individual Phases

```bash
# Phase 2: Linear systems (SDOF + 2DOF) — Section 3
python transformer_dynsys/run_all.py --phase 2

# Phase 3: Van der Pol oscillator — Section 4.1
python transformer_dynsys/run_all.py --phase 3

# Phase 4: Chafee-Infante PDE — Section 4.2
python transformer_dynsys/run_all.py --phase 4

# Phase 5: Navier-Stokes — Section 4.3
python transformer_dynsys/run_all.py --phase 5
```

---

## Model Architecture

The core architecture is a **single-layer, single-head Transformer** following the paper's Figure 1:

```
Input → [Linear Embedding] → Causal Self-Attention → [Linear | MLP] → ŷ
```

Key components:
- **Input embedding:** `x^{emb}(t) = x(t) W^{emb}`
- **Causal attention:** `Z = softmax(QK^T / √d_k) V` with autoregressive masking
- **Residual connection:** Optional `Z ← Z + X̃`
- **Output head:** Linear (Eq. 6) or MLP (Eq. 5) applied to the last token

---

## Experiments

### Section 3: Linear Dynamical Systems

| System | Description | Key Result |
|--------|-------------|------------|
| **SDOF** | Single degree-of-freedom oscillator | AR spectrum matches theoretical resonance |
| **2DOF** | Coupled oscillator (2 masses) | Vector AR spectrum captures both modes |

### Section 4: Nonlinear Dynamical Systems

| System | Description | Key Analysis |
|--------|-------------|--------------|
| **Van der Pol** | Limit cycle oscillator | Partial vs. full observability comparison |
| **Chafee-Infante** | Reaction-diffusion PDE | Latent space bifurcation structure |
| **Navier-Stokes** | 2D cylinder flow | Vortex shedding dynamics recovery |

---

## Configuration

Default hyperparameters are defined in `config/defaults.py`:

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-3 |
| Batch size | 64 |
| Epochs | 1000 |
| Seeds per config | 10 |

System-specific parameters (mass, damping, stiffness, etc.) match those in the paper.

---

## Analysis Tools

### Spectral Analysis

Extract learned AR coefficients and compare to theoretical spectra:

```python
from analysis.spectral import compute_learned_ar_spectrum_sdof

freqs, H_mag, ar_coeffs = compute_learned_ar_spectrum_sdof(
    model, X_sample, fs=25.0
)
```

### Attention Visualization

Visualize attention patterns across input tokens.

### Latent Space Analysis

Examine the learned embeddings and their relationship to the underlying dynamics.

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{duthe2025mechanistic,
  title={A Mechanistic Analysis of Transformers for Dynamical Systems},
  author={Duthé, et al.},
  journal={arXiv preprint arXiv:2512.21113},
  year={2025}
}
```

---

## License

This reproduction is provided for educational and research purposes.
