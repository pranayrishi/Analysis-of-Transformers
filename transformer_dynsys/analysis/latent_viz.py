"""
Latent-space visualisation helpers (Figures 5, 7, 8).

* Z vs x plots (1-D partial-observation)
* Z + x^{emb} phase portraits (2-D / 3-D inner dim)
* Q, K, V scatter plots
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_latent_1d(
    x_values, z_values, ax=None, label="", color="C0", ylabel="Z"
):
    """Plot Z (or Z+x) versus observed x for a 1-D latent model."""
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(x_values, z_values, s=2, alpha=0.5, c=color, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    if label:
        ax.legend(markerscale=4)
    return ax


def plot_latent_2d(
    z1, z2, ax=None, label="", color=None, colorbar_label=None, **kw
):
    """2-D scatter of latent components (e.g. Z_1+x^{emb}_1 vs Z_2+x^{emb}_2)."""
    if ax is None:
        _, ax = plt.subplots()
    sc = ax.scatter(z1, z2, s=2, alpha=0.5, c=color, cmap="viridis", label=label, **kw)
    ax.set_xlabel("$z_1 + x^{\\mathrm{emb}}_1$")
    ax.set_ylabel("$z_2 + x^{\\mathrm{emb}}_2$")
    if color is not None and colorbar_label:
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label(colorbar_label)
    if label:
        ax.legend(markerscale=4)
    return ax


def plot_latent_3d_projections(
    z_plus_emb, color_values, color_labels, axes=None, suptitle=""
):
    """Three 2-D projections of a 3-D latent space, each coloured by a
    different Fourier mode (Figure 8)."""
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(suptitle)

    pairs = [(0, 1), (0, 2), (1, 2)]
    for ax, (i, j), cval, clbl in zip(axes, pairs, color_values, color_labels):
        sc = ax.scatter(z_plus_emb[:, i], z_plus_emb[:, j], s=2,
                        alpha=0.4, c=cval, cmap="coolwarm")
        ax.set_xlabel(f"$z_{i+1}+x^{{\\mathrm{{emb}}}}_{i+1}$")
        ax.set_ylabel(f"$z_{j+1}+x^{{\\mathrm{{emb}}}}_{j+1}$")
        plt.colorbar(sc, ax=ax, label=clbl)

    return axes
